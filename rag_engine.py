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
from langchain.chains import RetrievalQA

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Cấu hình tập trung ────────────────────────────────────────
CONFIG = {
    "embedding_model":  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "embedding_device": "cpu",
    "chunk_size":       1000,   # default — bị override bởi tham số khi gọi process_pdf/process_docx
    "chunk_overlap":    100,    # default — bị override bởi tham số khi gọi process_pdf/process_docx
    "retriever_k":      5,
    "llm_model":        "qwen2.5:7b",
    "llm_temperature":  0.7,
    "ollama_host":      os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
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


# ══════════════════════════════════════════════════════════════
# 2. XỬ LÝ TÀI LIỆU
# ══════════════════════════════════════════════════════════════

def process_pdf(file_path: str, embedder, chunk_size: int = None, chunk_overlap: int = None) -> tuple:
    """
    Đọc PDF → chunk → FAISS → retriever.

    chunk_size / chunk_overlap: nếu truyền vào thì dùng, ngược lại lấy từ CONFIG.
    Trả về: (retriever, num_pages, num_chunks)
    """
    loader   = PDFPlumberLoader(file_path)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or CONFIG["chunk_overlap"],
    )
    chunks       = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embedder)
    retriever    = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["retriever_k"]},
    )
    return retriever, len(docs), len(chunks)


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
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embedder)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["retriever_k"]},
    )

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

    return retriever, num_pages, len(chunks)

# ══════════════════════════════════════════════════════════════
# 3. HELPERS NỘI BỘ
# ══════════════════════════════════════════════════════════════

def _detect_language(text: str) -> str:
    """Trả về 'vi' nếu phát hiện tiếng Việt, ngược lại 'en'."""
    return "vi" if any(c in text.lower() for c in _VIET_CHARS) else "en"


def _build_prompt(language: str) -> PromptTemplate:
    if language == "vi":
        template = """Bạn là trợ lý chuyên phân tích tài liệu. Nhiệm vụ của bạn là trả lời câu hỏi \
DỰA HOÀN TOÀN vào ngữ cảnh được cung cấp bên dưới.

QUY TẮC:
- Chỉ sử dụng thông tin có trong [NGỮ CẢNH]. Không thêm kiến thức bên ngoài.
- Nếu ngữ cảnh có đủ thông tin: trả lời đầy đủ, có cấu trúc rõ ràng.
- Nếu ngữ cảnh chỉ có một phần: trả lời phần biết, nói rõ phần nào không có trong tài liệu.
- Nếu ngữ cảnh không có thông tin liên quan: trả lời "Tài liệu không đề cập đến vấn đề này."
- Trích dẫn ý chính từ ngữ cảnh khi cần thiết để tăng độ tin cậy.

[NGỮ CẢNH]
{context}

[CÂU HỎI]
{question}

[TRẢ LỜI]"""

    else:
        template = """You are a document analysis assistant. Your task is to answer questions \
based EXCLUSIVELY on the context provided below.

RULES:
- Only use information present in [CONTEXT]. Do not add outside knowledge.
- If context has enough info: answer fully with clear structure.
- If context has partial info: answer what you can, clearly state what's missing.
- If context has no relevant info: respond "The document does not mention this topic."
- Quote key phrases from the context when it strengthens your answer.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]"""

    return PromptTemplate(template=template, input_variables=["context", "question"])


def _get_context(question: str, retriever) -> str:
    """Lấy context từ FAISS retriever."""
    docs = retriever.invoke(question)
    return "\n\n".join(d.page_content for d in docs)


# ══════════════════════════════════════════════════════════════
# 4. HỎI ĐÁP — STREAMING (hàm chính, dùng trong app.py)
# ══════════════════════════════════════════════════════════════

def ask_question_stream(question: str, retriever) -> Generator[str, None, None]:
    """
    Generator: yield từng token câu trả lời ngay khi LLM sinh ra.
    app.py dùng hàm này để hiển thị streaming trực tiếp lên UI.

    Cách dùng:
        for chunk in ask_question_stream(q, retriever):
            full_text += chunk
            placeholder.markdown(full_text + "▌")

    Mở rộng Câu 5 (citation): Sau vòng lặp yield, yield thêm
    "\n\n---\n**Nguồn:** ..." với thông tin từ source_documents.

    Mở rộng Câu 6 (Conversational RAG): Truyền thêm tham số
    chat_history và dùng ConversationalRetrievalChain.
    """
    lang    = _detect_language(question)
    prompt  = _build_prompt(lang)
    context = _get_context(question, retriever)

    # Điền prompt thủ công để dùng streaming trực tiếp với Ollama
    filled_prompt = prompt.format(context=context, question=question)

    llm = Ollama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_host"],
        temperature=CONFIG["llm_temperature"],
    )

    # stream=True → Ollama trả về iterator, mỗi item là 1 chunk văn bản
    for chunk in llm.stream(filled_prompt):
        yield chunk


# ══════════════════════════════════════════════════════════════
# 5. HỎI ĐÁP — NON-STREAMING (giữ lại để test / dùng ngoài UI)
# ══════════════════════════════════════════════════════════════

def ask_question(question: str, retriever) -> str:
    """
    Trả về toàn bộ câu trả lời dạng string (không stream).
    Dùng khi cần kết quả hoàn chỉnh, ví dụ: viết test, batch processing.
    """
    return "".join(ask_question_stream(question, retriever))

#xu ly cho cau 5
def ask_question_stream_with_sources(question: str, retriever, file_name: str = None) -> Generator[str,None, None]:
    """
    Giống ask_question_stream nhưng sau khi stream xong,
    yield thêm 1 item đặc biệt chứa source metadata.
    
    Convention: item cuối cùng bắt đầu bằng "@@SOURCES@@"
    app.py sẽ detect prefix này để tách ra xử lý riêng. 
    """
    lang = _detect_language(question)
    prompt = _build_prompt(lang)
    #lay source doc co metadata
    source_docs=retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in source_docs)

    filled_prompt = prompt.format(context=context, question=question)
    llm= Ollama(
        model= CONFIG["llm_model"],
        base_url= CONFIG["ollama_host"],
        temperature=CONFIG["llm_temperature"],
    )
    #stream
    for chunk in llm.stream(filled_prompt):
        yield chunk
     #Sau khi stream xong, đóng gói source info thành JSON
    import json
    sources=[]
    for i , doc in enumerate(source_docs):
        meta = doc.metadata # metadata la cai luu page,source
        sources.append({
            "index": i+1,
            "page": meta.get("page", "?"), #so trang
            "source":  file_name or os.path.basename(meta.get("source", "unknown")), #duong dan file
            "content": doc.page_content, #doan van goc
        })
    yield "@@SOURCES@@" + json.dumps(sources, ensure_ascii=False)
