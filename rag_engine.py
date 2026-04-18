import os
import re
import json
import networkx as nx
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
    "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "embedding_device": "cpu",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "retriever_k": 3,
    "llm_model": "qwen2.5:3b",
    "llm_temperature": 0.3,
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    "use_rerank": False,
    "use_self_rag": False,
    "self_rag_max_iter": 2,
    "selected_file": "All",
}

_VIET_CHARS = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"

# ══════════════════════════════════════════════════════════════
# 1. EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════

_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
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
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ══════════════════════════════════════════════════════════════
# 2. XỬ LÝ TÀI LIỆU
# ══════════════════════════════════════════════════════════════

def process_pdf(file_path: str, embedder, chunk_size: int = None, chunk_overlap: int = None) -> tuple:
    """Đọc PDF → chunk. Trả về: (chunks, num_pages, num_chunks)"""
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)
    return chunks, len(docs), len(chunks)


def process_docx(file_path: str, embedder, chunk_size: int = None, chunk_overlap: int = None) -> tuple:
    """Đọc DOCX → chunk. Trả về: (chunks, num_pages, num_chunks)"""
    from langchain_community.document_loaders import Docx2txtLoader
    import docx

    CHARS_PER_PAGE = 1000

    loader = Docx2txtLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

    try:
        doc = docx.Document(file_path)
        total_chars = sum(len(p.text.strip()) for p in doc.paragraphs if p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    total_chars += len(cell.text.strip())
        num_pages = max(1, round(total_chars / CHARS_PER_PAGE))
    except Exception:
        total_chars = sum(len(d.page_content) for d in docs)
        num_pages = max(1, round(total_chars / CHARS_PER_PAGE))

    return chunks, num_pages, len(chunks)


def build_hybrid_retriever(chunks, embedder):
    """Xây Hybrid Retriever (FAISS + BM25)."""
    vector_store = FAISS.from_documents(chunks, embedder)
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["retriever_k"]},
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = CONFIG["retriever_k"]

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )
    return hybrid_retriever


# ══════════════════════════════════════════════════════════════
# 3. GRAPH RAG — Xây đồ thị thực thể
# ══════════════════════════════════════════════════════════════

def _extract_entities_simple(text: str) -> list[str]:
    """
    Trích xuất thực thể từ text — generic, hỗ trợ mọi lĩnh vực.

    Chiến lược 2 lớp (bỏ hardcode domain-specific):
      1. Acronym / Title Case tiếng Anh — bắt tên riêng, viết tắt
      2. N-gram động (bigram + trigram) — bắt cụm khái niệm bất kỳ
         trong văn bản, không phụ thuộc lĩnh vực
    """
    entities = set()
    text_lower = text.lower()

    # ── Lớp 1: Tiếng Anh — Acronym & Title Case ─────────────
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    entities.update(acronyms)

    title_case = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    entities.update(title_case)

    # ── Lớp 2: N-gram động (bigram + trigram) ────────────────
    # Giữ nguyên dấu tiếng Việt
    words = re.findall(r'[\w\u00C0-\u024F\u1E00-\u1EFF]+', text_lower)

    _STOP = {
        # Tiếng Việt — stop words phổ biến
        "là", "của", "và", "các", "có", "cho", "trong", "với", "về",
        "được", "này", "đó", "khi", "nào", "như", "hay", "hoặc", "một",
        "những", "để", "theo", "từ", "tới", "đến", "trên", "dưới",
        "không", "thể", "cần", "phải", "sẽ", "đã", "đang", "rất",
        "hãy", "nêu", "trình", "bày", "liệt", "mọi", "bởi", "vì",
        "sau", "trước", "giữa", "tại", "qua", "còn", "lại", "nên",
        # Tiếng Anh — stop words
        "the", "of", "and", "for", "in", "is", "to", "a", "an",
        "it", "on", "at", "by", "or", "be", "that", "this", "with",
        "are", "was", "were", "has", "have", "had", "not", "but",
        "from", "they", "their", "can", "will", "its", "which",
    }
    filtered = [w for w in words if len(w) >= 3 and w not in _STOP]

    # Bigram — cụm 2 từ (min 7 ký tự để bỏ cặp quá ngắn)
    for i in range(len(filtered) - 1):
        bigram = f"{filtered[i]} {filtered[i+1]}"
        if len(bigram) >= 7:
            entities.add(bigram)

    # Trigram — cụm 3 từ (min 10 ký tự)
    for i in range(len(filtered) - 2):
        trigram = f"{filtered[i]} {filtered[i+1]} {filtered[i+2]}"
        if len(trigram) >= 10:
            entities.add(trigram)

    return list(entities)[:40]  # tăng lên 40 để graph phong phú hơn


def build_graph_rag(chunks: list) -> nx.Graph:
    """
    Xây đồ thị từ danh sách chunks:
    - Node: thực thể (entity) được trích xuất từ text
    - Edge: hai entity cùng xuất hiện trong 1 chunk → có liên hệ
    - Node attribute: list các chunk_id chứa entity đó
    - Edge attribute: số lần đồng xuất hiện (co-occurrence weight)

    Trả về: networkx.Graph
    """
    G = nx.Graph()

    for idx, chunk in enumerate(chunks):
        text = chunk.page_content
        page = chunk.metadata.get("page", "?")
        source = chunk.metadata.get("source", "unknown")

        entities = _extract_entities_simple(text)

        # Thêm node cho từng entity
        for ent in entities:
            if G.has_node(ent):
                G.nodes[ent]["chunks"].append({
                    "chunk_idx": idx,
                    "page": page,
                    "source": source,
                    "content": text,
                })
            else:
                G.add_node(ent, chunks=[{
                    "chunk_idx": idx,
                    "page": page,
                    "source": source,
                    "content": text,
                }])

        # Thêm edge giữa các entity trong cùng chunk
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    return G


def _score_chunk(content: str, keywords: list) -> float:
    """
    Tính relevance score của một chunk dựa trên 3 tín hiệu:
      - coverage  : tỉ lệ keywords xuất hiện trong chunk     [0,1]
      - hits_norm : số hits normalized theo tổng keywords    [0,1]
      - density   : mật độ keyword so với độ dài chunk       [0,1]
    Trả về float trong [0, 1].
    """
    if not keywords or not content:
        return 0.0
    content_lower = content.lower()
    word_count = max(len(content_lower.split()), 1)
    hits = sum(1 for kw in keywords if kw in content_lower)
    if hits == 0:
        return 0.0
    coverage       = hits / len(keywords)
    hits_norm      = min(hits / len(keywords), 1.0)
    density        = min(hits / (word_count / 50), 1.0)   # ~1 hit/50 words = max density
    return 0.5 * coverage + 0.3 * hits_norm + 0.2 * density


def _graph_retrieve(question: str, graph: nx.Graph, top_k: int = 6) -> list[dict]:
    """
    Scored retrieval — gán điểm tổng hợp cho mỗi chunk từ 3 tín hiệu:

      Signal 1 — Direct node match (base_score = match_strength ∈ [0.7, 1.0]):
        Entity/keyword từ câu hỏi khớp trực tiếp với node trong graph.
        Exact match → 1.0, partial match → 0.7–0.9 (tỉ lệ overlap).

      Signal 2 — Neighbor expansion (base_score ≤ 0.5):
        Chỉ lấy tối đa MAX_NEIGHBORS neighbor per matched node,
        weighted theo edge_weight (co-occurrence thực sự cao mới lấy).
        base_score = node_strength × 0.5 × (edge_w / max_edge_w).

      Signal 3 — Keyword relevance bonus (cộng vào mọi chunk, tối đa +0.5):
        _score_chunk() đo coverage + density của keywords trong content.

    final_score = base_score + 0.5 × keyword_score  ∈ [0, 1.5]

    Chỉ giữ chunk có final_score ≥ MIN_SCORE (0.15) để lọc nhiễu.
    Fallback: keyword-only search toàn graph nếu không có chunk nào pass.
    """
    MIN_SCORE    = 0.15   # ngưỡng lọc: chunk dưới mức này bị bỏ
    MAX_NEIGHBORS = 2     # tối đa neighbor per matched node

    # ── Chuẩn bị keywords ─────────────────────────────────────
    _STOP = {
        "là", "của", "và", "các", "có", "cho", "trong", "với", "về",
        "được", "này", "đó", "khi", "nào", "như", "hay", "hoặc", "một",
        "những", "để", "theo", "hãy", "nêu", "trình", "bày", "liệt", "kê",
        "từng", "mỗi", "tất", "cả", "luôn", "đều", "sao", "vậy", "thế",
        "the", "of", "and", "for", "in", "is", "to", "a", "an", "what",
        "how", "why", "when", "which", "who", "does", "do", "are", "was",
    }
    keywords = [
        w for w in re.findall(r'[\w\u00C0-\u024F\u1E00-\u1EFF]+', question.lower())
        if len(w) >= 3 and w not in _STOP
    ]

    # ── Signal 1: Direct node match ───────────────────────────
    q_entities = _extract_entities_simple(question)
    raw_words = [w for w in re.findall(r'[\w\u00C0-\u024F\u1E00-\u1EFF]+', question.lower())
                 if len(w) >= 4]
    q_entities = list(set(q_entities + raw_words))

    matched_nodes: dict = {}   # node → match_strength
    for ent in q_entities:
        ent_lower = ent.lower()
        if graph.has_node(ent):
            matched_nodes[ent] = 1.0
        elif graph.has_node(ent_lower):
            matched_nodes[ent_lower] = 1.0
        else:
            for node in graph.nodes():
                node_lower = node.lower()
                # Partial match: chỉ lấy nếu overlap đủ dài (≥ 4 ký tự)
                if ent_lower in node_lower:
                    overlap = ent_lower
                elif node_lower in ent_lower:
                    overlap = node_lower
                else:
                    continue
                if len(overlap) >= 4:
                    strength = len(overlap) / max(len(ent_lower), len(node_lower))
                    if node not in matched_nodes or matched_nodes[node] < strength:
                        matched_nodes[node] = min(strength, 0.9)

    # ── Signal 2: Neighbor expansion (chọn lọc) ───────────────
    neighbor_nodes: dict = {}  # node → neighbor_score
    for node, node_strength in matched_nodes.items():
        nbrs = list(graph.neighbors(node))
        if not nbrs:
            continue
        max_w = max(graph[node][n].get("weight", 1) for n in nbrs)
        nbrs.sort(key=lambda n: graph[node][n].get("weight", 0), reverse=True)
        for nb in nbrs[:MAX_NEIGHBORS]:
            if nb in matched_nodes:   # không ghi đè direct match
                continue
            edge_w  = graph[node][nb].get("weight", 1)
            nb_score = node_strength * 0.5 * (edge_w / max_w)
            if nb not in neighbor_nodes or neighbor_nodes[nb] < nb_score:
                neighbor_nodes[nb] = nb_score

    # ── Gom chunks + tính final_score ─────────────────────────
    chunk_scores: dict = {}   # chunk_idx → (final_score, chunk_data)

    for node, strength in matched_nodes.items():
        if not graph.has_node(node):
            continue
        for ci in graph.nodes[node].get("chunks", []):
            idx = ci["chunk_idx"]
            kw  = _score_chunk(ci["content"], keywords)
            final = strength + 0.5 * kw
            if idx not in chunk_scores or chunk_scores[idx][0] < final:
                chunk_scores[idx] = (final, ci)

    for node, nb_score in neighbor_nodes.items():
        if not graph.has_node(node):
            continue
        for ci in graph.nodes[node].get("chunks", []):
            idx = ci["chunk_idx"]
            kw  = _score_chunk(ci["content"], keywords)
            final = nb_score + 0.5 * kw
            if idx not in chunk_scores or chunk_scores[idx][0] < final:
                chunk_scores[idx] = (final, ci)

    # ── Lọc MIN_SCORE + sort giảm dần ─────────────────────────
    ranked = sorted(
        [(s, ci) for s, ci in chunk_scores.values() if s >= MIN_SCORE],
        key=lambda x: x[0],
        reverse=True,
    )
    if ranked:
        return [ci for _, ci in ranked[:top_k]]

    # ── Fallback: keyword-only search toàn graph ──────────────
    fallback: list = []
    seen_fb: set = set()
    for node in graph.nodes():
        for ci in graph.nodes[node].get("chunks", []):
            idx = ci["chunk_idx"]
            if idx in seen_fb:
                continue
            seen_fb.add(idx)
            s = _score_chunk(ci["content"], keywords)
            if s > 0:
                fallback.append((s, ci))
    fallback.sort(key=lambda x: x[0], reverse=True)
    return [ci for _, ci in fallback[:top_k]]


# ══════════════════════════════════════════════════════════════
# 4. HELPERS NỘI BỘ
# ══════════════════════════════════════════════════════════════

def _detect_language(text: str) -> str:
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

[HISTORY]
{chat_history}

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]"""

    return PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])


def _build_graph_prompt(language: str) -> PromptTemplate:
    """
    Prompt riêng cho Graph RAG — nhấn mạnh khai thác quan hệ giữa các thực thể.
    Yêu cầu mô tả MỐI LIÊN HỆ, SO SÁNH và trả lời ĐẦY ĐỦ hơn Classic RAG.
    """
    if language == "vi":
        template = """Bạn là trợ lý phân tích tài liệu chuyên sâu, sử dụng Graph RAG.
Các đoạn văn dưới đây được chọn dựa trên QUAN HỆ giữa các thực thể trong tài liệu.

NHIỆM VỤ: Trả lời câu hỏi dựa trên ngữ cảnh, đặc biệt chú ý:
- Mô tả MỐI LIÊN HỆ giữa các khái niệm/thực thể khi có thể
- Giải thích TẠI SAO các thực thể liên quan đến nhau
- So sánh, đối chiếu nếu có nhiều thực thể cùng loại
- Trả lời ĐẦY ĐỦ và CHI TIẾT, có cấu trúc rõ ràng (dùng gạch đầu dòng hoặc đánh số nếu cần)

QUY TẮC:
- Chỉ dùng thông tin trong [NGỮ CẢNH]. Không thêm kiến thức ngoài.
- Nếu ngữ cảnh đủ thông tin: PHÂN TÍCH SÂU, không trả lời ngắn.
- Nếu ngữ cảnh thiếu: trả lời phần có, ghi rõ phần không có trong tài liệu.
- Nếu không có thông tin: "Tài liệu không đề cập đến vấn đề này."
- Ưu tiên liên kết thông tin từ NHIỀU đoạn khác nhau.

[HỘI THOẠI TRƯỚC]
{chat_history}

[NGỮ CẢNH — {num_chunks} đoạn liên quan]
{context}

[CÂU HỎI]
{question}

[TRẢ LỜI — Phân tích đầy đủ, có cấu trúc]"""
    else:
        template = """You are a document analysis assistant using Graph RAG.
The passages below are selected based on RELATIONSHIPS between entities in the document.

TASK: Answer the question with special attention to:
- Describing RELATIONSHIPS between concepts/entities when possible
- Explaining WHY entities are related
- Comparing and contrasting when multiple similar entities exist
- Giving COMPLETE, DETAILED answers with clear structure (use bullets or numbering)

RULES:
- Only use information in [CONTEXT]. No outside knowledge.
- If context is sufficient: give a DEEP ANALYSIS, not a short answer.
- If context is partial: answer what you can, clearly note what's missing.
- If no relevant info: "The document does not mention this topic."
- Prioritize linking information ACROSS multiple passages.

[HISTORY]
{chat_history}

[CONTEXT — {num_chunks} relevant passages]
{context}

[QUESTION]
{question}

[ANSWER — Full structured analysis]"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history", "num_chunks"]
    )


def _format_chat_history(chat_history: list, max_turns: int = 3) -> str:
    if not chat_history:
        return ""
    history_text = []
    for item in chat_history[-max_turns:]:
        history_text.append(f"User: {item.get('question', '')}")
        history_text.append(f"Assistant: {item.get('answer', '')}")
    return "\n".join(history_text)


def rerank_documents(question: str, docs: list, top_k: int = 5):
    if not docs:
        return docs
    cross_encoder = get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
    docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    return docs[:top_k]


def _get_llm():
    return Ollama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_host"],
        temperature=CONFIG["llm_temperature"],
    )


# ══════════════════════════════════════════════════════════════
# 5. HỎI ĐÁP — CLASSIC RAG (streaming)
# ══════════════════════════════════════════════════════════════

def ask_question_stream_with_sources(
    question: str,
    retriever,
    chat_history: list = None,
    file_name: str = None,
) -> Generator[str, None, None]:

    lang = _detect_language(question)
    prompt = _build_prompt(lang)
    history_text = _format_chat_history(chat_history or [])

    # 1. Lấy đủ candidates để rerank sau này
    retrieved_docs = retriever.invoke(question)[:20]

    # 2. Áp dụng filter selected_file (nếu người dùng chọn 1 file cụ thể)
    selected_file = CONFIG.get("selected_file", "All")
    if selected_file != "All":
        retrieved_docs = [
            d for d in retrieved_docs 
            if d.metadata.get("source") == selected_file
        ]

    # 3. Self-RAG path (giữ nguyên)
    if CONFIG.get("use_self_rag", False):
        answer, score, context, docs = self_rag_pipeline(question, retriever, chat_history)
        yield answer
        sources = [
            {
                "index": i + 1,
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content,
            }
            for i, doc in enumerate(docs)
        ]
        yield "@@SOURCES@@" + json.dumps(sources, ensure_ascii=False)
        yield "@@CONFIDENCE@@" + str(score)
        return

    # 4. Rerank hoặc lấy đúng top-k
    if CONFIG.get("use_rerank", False):
        source_docs = rerank_documents(
            question, 
            retrieved_docs, 
            top_k=CONFIG["retriever_k"]
        )
    else:
        source_docs = retrieved_docs[:CONFIG["retriever_k"]]

    # Xây context và prompt
    context = "\n\n".join(d.page_content for d in source_docs)
    filled_prompt = prompt.format(
        context=context, 
        question=question, 
        chat_history=history_text
    )

    # Stream câu trả lời
    llm = _get_llm()
    for chunk in llm.stream(filled_prompt):
        for c in chunk:
            yield c

    # Trả sources
    sources = [
        {
            "index": i + 1,
            "page": meta.get("page", "?"),
            "source": file_name or os.path.basename(meta.get("source", "unknown")),
            "content": doc.page_content,
        }
        for i, doc in enumerate(source_docs)
        for meta in [doc.metadata]
    ]
    yield "@@SOURCES@@" + json.dumps(sources, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# 6. HỎI ĐÁP — GRAPH RAG (non-streaming, trả về string)
# ══════════════════════════════════════════════════════════════

def ask_graph_rag(
    question: str,
    graph: nx.Graph,
    chat_history: list = None,
) -> tuple[str, list]:
    """
    Trả lời câu hỏi bằng Graph RAG.
    Trả về: (answer_string, sources_list)
    """
    lang = _detect_language(question)
    prompt = _build_graph_prompt(lang)
    history_text = _format_chat_history(chat_history or [])

    graph_chunks = _graph_retrieve(question, graph, top_k=CONFIG["retriever_k"])

    if not graph_chunks:
        return "Graph RAG không tìm thấy thông tin liên quan trong đồ thị.", []

    context = "\n\n".join(c["content"] for c in graph_chunks)
    filled_prompt = prompt.format(
        context=context,
        question=question,
        chat_history=history_text,
        num_chunks=len(graph_chunks),
    )

    llm = _get_llm()
    answer = llm.invoke(filled_prompt)

    sources = [
        {
            "index": i + 1,
            "page": c.get("page", "?"),
            "source": os.path.basename(str(c.get("source", "unknown"))),
            "content": c["content"],
        }
        for i, c in enumerate(graph_chunks)
    ]

    return answer, sources


# ══════════════════════════════════════════════════════════════
# 7. SO SÁNH CLASSIC RAG vs GRAPH RAG (generator)
# ══════════════════════════════════════════════════════════════

def ask_compare_rag(
    question: str,
    retriever,
    graph: nx.Graph,
    chat_history: list = None,
) -> Generator[str, None, None]:
    """
    Generator yield các token theo protocol:
      @@CLASSIC_START@@         — bắt đầu classic rag answer
      <token>...                — từng token của classic answer (streaming)
      @@CLASSIC_SOURCES@@ JSON  — sources của classic
      @@GRAPH_START@@           — bắt đầu graph rag answer
      <token>...                — từng token của graph answer (streaming)
      @@GRAPH_SOURCES@@ JSON    — sources của graph
      @@COMPARE_DONE@@          — kết thúc
    """
    lang = _detect_language(question)
    history_text = _format_chat_history(chat_history or [])
    llm = _get_llm()

    # ── Phần 1: Classic RAG (stream từng token) ───────────────
    yield "@@CLASSIC_START@@"

    classic_prompt = _build_prompt(lang)
    source_docs = retriever.invoke(question)[:CONFIG["retriever_k"]]
    if CONFIG.get("use_rerank", False):
        source_docs = rerank_documents(question, source_docs, top_k=CONFIG["retriever_k"])

    context_classic = "\n\n".join(d.page_content for d in source_docs)
    filled_classic = classic_prompt.format(
        context=context_classic, question=question, chat_history=history_text
    )

    for chunk in llm.stream(filled_classic):
        for c in chunk:
            yield c

    classic_sources = [
        {
            "index": i + 1,
            "page": doc.metadata.get("page", "?"),
            "source": os.path.basename(str(doc.metadata.get("source", "unknown"))),
            "content": doc.page_content,
        }
        for i, doc in enumerate(source_docs)
    ]
    yield "@@CLASSIC_SOURCES@@" + json.dumps(classic_sources, ensure_ascii=False)

    # ── Phần 2: Graph RAG (stream từng token + prompt riêng) ──
    yield "@@GRAPH_START@@"

    graph_top_k = max(CONFIG["retriever_k"], 6)  # Graph RAG cần nhiều chunks hơn
    graph_chunks = _graph_retrieve(question, graph, top_k=graph_top_k)

    if graph_chunks:
        graph_prompt = _build_graph_prompt(lang)
        context_graph = "\n\n".join(c["content"] for c in graph_chunks)
        filled_graph = graph_prompt.format(
            context=context_graph,
            question=question,
            chat_history=history_text,
            num_chunks=len(graph_chunks),
        )
        for chunk in llm.stream(filled_graph):
            for c in chunk:
                yield c
    else:
        no_info = "Graph RAG không tìm thấy thông tin liên quan trong đồ thị."
        for c in no_info:
            yield c
        graph_chunks = []

    graph_sources = [
        {
            "index": i + 1,
            "page": c.get("page", "?"),
            "source": os.path.basename(str(c.get("source", "unknown"))),
            "content": c["content"],
        }
        for i, c in enumerate(graph_chunks)
    ]
    yield "@@GRAPH_SOURCES@@" + json.dumps(graph_sources, ensure_ascii=False)
    yield "@@COMPARE_DONE@@"


# ══════════════════════════════════════════════════════════════
# 8. SELF-RAG
# ══════════════════════════════════════════════════════════════

def rewrite_query(question: str, chat_history: str = "") -> str:
    llm = _get_llm()
    prompt = f"""Rewrite the question to be clearer and more specific for document retrieval.

Chat History:
{chat_history}

Original Question:
{question}

Rewritten Question:"""
    return llm.invoke(prompt).strip()


def evaluate_answer(question: str, answer: str, context: str) -> dict:
    llm = Ollama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_host"],
        temperature=0.0,
    )
    prompt = f"""Evaluate the answer quality.

Question: {question}
Context: {context}
Answer: {answer}

Score from 0 to 1. Return JSON only:
{{"score": ..., "reason": "..."}}"""
    result = llm.invoke(prompt)
    try:
        return json.loads(result)
    except Exception:
        return {"score": 0.5, "reason": "parse_error"}


def self_rag_pipeline(question, retriever, chat_history=None):
    history_text = _format_chat_history(chat_history or [])
    best_answer, best_score, best_context, best_docs = "", 0, "", []

    for _ in range(CONFIG["self_rag_max_iter"]):
        new_query = rewrite_query(question, history_text)

        # ====================== ĐỒNG BỘ VỚI RAG THƯỜNG ======================
        # 1. Lấy đủ candidates (top-20) để rerank
        retrieved_docs = retriever.invoke(new_query)[:20]

        # 2. Áp dụng filter selected_file (nếu người dùng chọn 1 file cụ thể)
        selected_file = CONFIG.get("selected_file", "All")
        if selected_file != "All":
            retrieved_docs = [
                d for d in retrieved_docs 
                if d.metadata.get("source") == selected_file
            ]

        # 3. Rerank hoặc lấy đúng top-k theo giao diện
        if CONFIG.get("use_rerank", False):
            docs = rerank_documents(
                new_query, 
                retrieved_docs, 
                top_k=CONFIG["retriever_k"]
            )
        else:
            docs = retrieved_docs[:CONFIG["retriever_k"]]


        context = "\n\n".join(d.page_content for d in docs)
        prompt = _build_prompt(_detect_language(question))
        filled = prompt.format(context=context, question=question, chat_history=history_text)

        llm = _get_llm()
        answer = llm.invoke(filled)
        eval_result = evaluate_answer(question, answer, context)
        score = eval_result.get("score", 0)

        if score > best_score:
            best_score, best_answer, best_context, best_docs = score, answer, context, docs

        if score > 0.8:
            break

    return best_answer, best_score, best_context, best_docs