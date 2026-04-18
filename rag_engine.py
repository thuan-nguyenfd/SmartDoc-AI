# ══════════════════════════════════════════════════════════════
# rag_engine.py  [v1.3 - streaming]
# Chịu trách nhiệm: Toàn bộ logic AI — embedding, FAISS, LLM, prompt
#
# File này KHÔNG biết Streamlit là gì.
# Nó chỉ nhận tham số → xử lý → trả về kết quả.
#
# Thay đổi v1.3:
#   - Thêm ask_question_stream() — generator yield từng token
#     để app.py có thể streaming trực tiếp ra UI
#   - Giữ nguyên ask_question() cho trường hợp không cần stream
#
# Để mở rộng (Câu 1, 4, 5, 6, 7, 9, 10 trong mục 8):
#   Câu 1 (DOCX)       : Thêm hàm process_docx() bên dưới process_pdf()
#   Câu 4 (chunk)      : Sửa CHUNK_SIZE / CHUNK_OVERLAP trong CONFIG
#   Câu 5 (citation)   : Sửa ask_question_stream() để yield thêm sources
#   Câu 6 (conv. RAG)  : Thêm ConversationalRetrievalChain
#   Câu 7 (hybrid)     : Thêm BM25Retriever + EnsembleRetriever
#   Câu 9 (re-ranking) : Thêm CrossEncoderReranker sau retriever
# ══════════════════════════════════════════════════════════════

import os
from typing import Generator

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Cấu hình tập trung ────────────────────────────────────────
CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_device": "cpu",
    "chunk_size": 1000,  # default — bị override bởi tham số khi gọi process_pdf/process_docx
    "chunk_overlap": 100,  # default — bị override bởi tham số khi gọi process_pdf/process_docx
    "retriever_k": 3,
    "llm_model": "qwen2.5:1.5b",
    "llm_temperature": 0.3,
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    "use_rerank": False,
    "use_self_rag": False,
    "self_rag_max_iter": 2,
}

_VIET_CHARS = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"

# ══════════════════════════════════════════════════════════════
# 1. EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════

_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Trả về embedding model (lazy-load, chỉ khởi tạo 1 lần)."""
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding_model"],
            model_kwargs={"device": CONFIG["embedding_device"]},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _cross_encoder


# ══════════════════════════════════════════════════════════════
# 2. XỬ LÝ TÀI LIỆU
# ══════════════════════════════════════════════════════════════

def process_pdf(file_path: str, embedder, chunk_size: int = None, chunk_overlap: int = None) -> tuple:
    """
    Đọc PDF → chunk → FAISS → retriever.

    chunk_size / chunk_overlap: nếu truyền vào thì dùng, ngược lại lấy từ CONFIG.
    Trả về: (retriever, num_pages, num_chunks)
    """
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)
    return chunks, len(docs), len(chunks)


# Câu 1 — hỗ trợ DOCX:
def process_docx(file_path: str, embedder, chunk_size: int = None, chunk_overlap: int = None) -> tuple:
    """
    Đọc file DOCX → chunk → FAISS.

    chunk_size / chunk_overlap: nếu truyền vào thì dùng, ngược lại lấy từ CONFIG.
    Ước lượng số trang dựa trên tổng ký tự nội dung:
    - Trung bình 1 trang A4 ≈ 1800–2200 ký tự (font 12, giãn dòng 1.5)
    - Dùng 2000 ký tự/trang làm baseline, có thể chỉnh qua CHARS_PER_PAGE
    """
    from langchain_community.document_loaders import Docx2txtLoader
    import docx

    CHARS_PER_PAGE = 1000  # ← Chỉnh hằng số này nếu muốn calibrate

    loader = Docx2txtLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

    # ── Ước lượng số trang ────────────────────────────────────
    try:
        doc = docx.Document(file_path)

        # Đếm ký tự thực từ tất cả paragraph (bỏ khoảng trắng thừa)
        total_chars = sum(
            len(para.text.strip())
            for para in doc.paragraphs
            if para.text.strip()
        )

        # Cộng thêm ký tự trong bảng (table cells)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    total_chars += len(cell.text.strip())

        num_pages = max(1, round(total_chars / CHARS_PER_PAGE))

    except Exception:
        # Fallback: ước lượng từ tổng nội dung đã load
        total_chars = sum(len(d.page_content) for d in docs)
        num_pages = max(1, round(total_chars / CHARS_PER_PAGE))

    return chunks, num_pages, len(chunks)


def build_hybrid_retriever(chunks, embedder):

    # ── Vector search (semantic) ─────────────────────────────
    vector_store = FAISS.from_documents(chunks, embedder)
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["retriever_k"]},
    )

    # ── BM25 (keyword search) ────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = CONFIG["retriever_k"]

    # ── Ensemble (Hybrid) ───────────────────────────────────
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]  # có thể chỉnh
    )

    return hybrid_retriever


# ══════════════════════════════════════════════════════════════
# 3. HELPERS NỘI BỘ
# ══════════════════════════════════════════════════════════════

def _detect_language(text: str) -> str:
    """Trả về 'vi' nếu phát hiện tiếng Việt, ngược lại 'en'."""
    return "vi" if any(c in text.lower() for c in _VIET_CHARS) else "en"


def _build_prompt(language: str) -> PromptTemplate:
    if language == "vi":
        template = """Bạn là trợ lý chuyên phân tích tài liệu. 
Nhiệm vụ của bạn là: 
- Trả lời câu hỏi DỰA HOÀN TOÀN vào ngữ cảnh được cung cấp bên dưới.

QUY TẮC:
- Chỉ sử dụng thông tin có trong [NGỮ CẢNH]. Không thêm kiến thức bên ngoài.
- Nếu ngữ cảnh có đủ thông tin: trả lời đầy đủ, có cấu trúc rõ ràng.
- Nếu ngữ cảnh chỉ có một phần: trả lời phần biết, nói rõ phần nào không có trong tài liệu.
- Nếu ngữ cảnh không có thông tin liên quan: trả lời "Tài liệu không đề cập đến vấn đề này."
- Trích dẫn ý chính từ ngữ cảnh khi cần thiết để tăng độ tin cậy.
- Nếu câu hỏi là follow-up, hãy suy luận từ [HỘI THOẠI TRƯỚC]
- Có thể sử dụng [HỘI THOẠI TRƯỚC] để hiểu câu hỏi

[HỘI THOẠI TRƯỚC]
{chat_history}

[NGỮ CẢNH]
{context}

[CÂU HỎI]
{question}

[TRẢ LỜI]"""

    else:
        template = """You are a document analysis assistant. 
Your task is:
- Answer the question BASED ENTIRELY ON the [CONTEXT] provided below.

RULES:
- Only use information present in [CONTEXT]. Do not add outside knowledge.
- If context has enough info: answer fully with clear structure.
- If context has partial info: answer what you can, clearly state what's missing.
- If context has no relevant info: respond "The document does not mention this topic."
- Quote key phrases from the context when it strengthens your answer.
- If the question is follow-up, infer from the [HISTORY].
- You can use [HISTORY] to understand the question.

[HISTORY]
{chat_history}

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]"""

    return PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])


def _format_chat_history(chat_history: list, max_turns: int = 3) -> str:
    if not chat_history:
        return ""

    history_text = []
    for item in chat_history[-max_turns:]:
        q = item.get("question", "")
        a = item.get("answer", "")
        history_text.append(f"User: {q}")
        history_text.append(f"Assistant: {a}")

    return "\n".join(history_text)


def _get_context(question: str, retriever) -> str:
    docs = retriever.invoke(question)
    return "\n\n".join(d.page_content for d in docs)


def rerank_documents(question: str, docs: list, top_k: int = 5):
    if not docs:
        return docs

    cross_encoder = get_cross_encoder()

    # Tạo pairs (question, doc)
    pairs = [(question, doc.page_content) for doc in docs]

    # Predict relevance score
    scores = cross_encoder.predict(pairs)

    # Gán score vào doc
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)

    # Sort theo score giảm dần
    docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)

    return docs[:top_k]


# ══════════════════════════════════════════════════════════════
# 4. HỎI ĐÁP — NON-STREAMING (giữ lại để test / dùng ngoài UI)
# ══════════════════════════════════════════════════════════════


# xu ly cho cau 5
def ask_question_stream_with_sources(question: str, retriever, chat_history: list = None, file_name: str = None) -> Generator[str, None, None]:

    lang = _detect_language(question)
    prompt = _build_prompt(lang)
    history_text = _format_chat_history(chat_history or [])
    # lay source doc co metadata
    source_docs = retriever.invoke(question)[:10]

    selected_file = CONFIG.get("selected_file", "All")

    if selected_file != "All":
        source_docs = [
            d for d in source_docs
            if d.metadata.get("source") == selected_file
        ]

    if CONFIG.get("use_self_rag", False):
        answer, score, context, docs = self_rag_pipeline(
            question, retriever, chat_history
        )

        yield answer

        # ⭐ thêm sources
        import json
        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                "index": i + 1,
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content
            })

        yield "@@SOURCES@@" + json.dumps(sources, ensure_ascii=False)
        yield "@@CONFIDENCE@@" + str(score)
        return

    if CONFIG.get("use_rerank", False):
        source_docs = rerank_documents(question, source_docs, top_k=CONFIG["retriever_k"])

    context = "\n\n".join(d.page_content for d in source_docs)

    filled_prompt = prompt.format(context=context, question=question, chat_history=history_text)
    llm = Ollama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_host"],
        temperature=CONFIG["llm_temperature"],
    )
    # stream
    for chunk in llm.stream(filled_prompt):
        for c in chunk:
            yield c
    # Sau khi stream xong, đóng gói source info thành JSON
    import json
    sources = []
    for i, doc in enumerate(source_docs):
        meta = doc.metadata  # metadata la cai luu page,source
        sources.append({
            "index": i + 1,
            "page": meta.get("page", "?"),  # so trang
            "source": file_name or os.path.basename(meta.get("source", "unknown")),  # duong dan file
            "content": doc.page_content,  # doan van goc
        })
    yield "@@SOURCES@@" + json.dumps(sources, ensure_ascii=False)


def rewrite_query(question: str, chat_history: str = "") -> str:
    llm = Ollama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_host"],
        temperature=0.3,
    )

    prompt = f"""
Rewrite the question to be clearer and more specific for document retrieval.

Chat History:
{chat_history}

Original Question:
{question}

Rewritten Question:
"""

    response = llm.invoke(prompt)
    return response.strip()


def evaluate_answer(question: str, answer: str, context: str) -> dict:
    llm = Ollama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_host"],
        temperature=0.0,
    )

    prompt = f"""
Evaluate the answer quality.

Question: {question}
Context: {context}
Answer: {answer}

Score from 0 to 1:
Also explain briefly.

Return JSON:
{{"score": ..., "reason": "..."}}
"""

    result = llm.invoke(prompt)

    import json
    try:
        return json.loads(result)
    except:
        return {"score": 0.5, "reason": "parse_error"}


def self_rag_pipeline(question, retriever, chat_history=None):
    history_text = _format_chat_history(chat_history or [])

    best_answer = ""
    best_score = 0
    best_context = ""
    best_docs = []

    for _ in range(CONFIG["self_rag_max_iter"]):

        new_query = rewrite_query(question, history_text)

        docs = retriever.invoke(new_query)[:10]

        if CONFIG.get("use_rerank", False):
            docs = rerank_documents(new_query, docs)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = _build_prompt(_detect_language(question))
        filled = prompt.format(
            context=context,
            question=question,
            chat_history=history_text
        )

        llm = Ollama(
            model=CONFIG["llm_model"],
            base_url=CONFIG["ollama_host"],
            temperature=CONFIG["llm_temperature"],
        )

        answer = llm.invoke(filled)

        eval_result = evaluate_answer(question, answer, context)
        score = eval_result.get("score", 0)

        if score > best_score:
            best_score = score
            best_answer = answer
            best_context = context
            best_docs = docs

        if score > 0.8:
            break

    return best_answer, best_score, best_context, best_docs