import os
import re
import json
import numpy as np
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
# 3. GRAPH RAG — Xây đồ thị thực thể bằng LLM + Embedding Retrieval
# ══════════════════════════════════════════════════════════════

def _extract_entities_and_relations_llm(text: str, llm) -> list[dict]:
    """
    Dùng LLM (Qwen) để trích xuất entities và relationships có ngữ nghĩa thực sự.

    Trả về list of {"e1": str, "rel": str, "e2": str}
    Ví dụ: {"e1": "FAISS", "rel": "lưu trữ", "e2": "vector embeddings"}

    Fallback về regex nếu LLM parse thất bại.
    """
    prompt = f"""Extract key entities and their relationships from the text below.
Return ONLY a JSON array, no explanation, no markdown.
Format: [{{"e1": "entity1", "rel": "relationship", "e2": "entity2"}}]
Rules:
- e1 and e2 must be specific concepts, tools, methods, or named things (not generic words)
- rel must be a short verb phrase describing how e1 relates to e2
- Extract 3 to 8 relationships maximum
- If text is in Vietnamese, keep entities in Vietnamese

Text:
{text[:600]}

JSON:"""
    try:
        raw = llm.invoke(prompt).strip()
        # Bỏ markdown fence nếu có
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        relations = json.loads(raw)
        # Validate: phải là list of dict với 3 keys
        valid = [
            r for r in relations
            if isinstance(r, dict)
            and all(k in r for k in ("e1", "rel", "e2"))
            and r["e1"].strip() and r["e2"].strip()
        ]
        return valid[:8]
    except Exception:
        # Fallback: chỉ extract acronym + Title Case làm entities đơn
        entities = list(set(
            re.findall(r'\b[A-Z]{2,}\b', text) +
            re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        ))[:6]
        relations = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                relations.append({"e1": entities[i], "rel": "related_to", "e2": entities[j]})
        return relations[:8]


def build_graph_rag(chunks: list) -> nx.Graph:
    """
    Xây Knowledge Graph từ chunks bằng LLM-based entity extraction.

    Kiến trúc chuẩn Graph RAG (Microsoft 2024):
    - Node: entity thực sự (tên công cụ, khái niệm, phương pháp)
    - Edge: relationship có ngữ nghĩa ("sử dụng", "là thành phần của", ...)
    - Node attribute:
        * chunks  : list các đoạn văn chứa entity này
        * embedding: vector của entity name (để semantic search)
    - Edge attribute:
        * rel     : tên quan hệ
        * weight  : số lần quan hệ này xuất hiện

    Retrieve sau này dùng embedding similarity thay vì keyword match.
    """
    llm = _get_llm()
    embedder = get_embedder()
    G = nx.Graph()

    # Batch size nhỏ để không quá tải Qwen 3b
    BATCH = 1

    for idx, chunk in enumerate(chunks):
        text = chunk.page_content
        page = chunk.metadata.get("page", "?")
        source = chunk.metadata.get("source", "unknown")
        chunk_data = {
            "chunk_idx": idx,
            "page": page,
            "source": source,
            "content": text,
        }

        relations = _extract_entities_and_relations_llm(text, llm)

        for rel in relations:
            e1 = rel["e1"].strip()
            e2 = rel["e2"].strip()
            relationship = rel["rel"].strip()

            # ── Thêm / cập nhật node e1 ──────────────────────
            if G.has_node(e1):
                G.nodes[e1]["chunks"].append(chunk_data)
            else:
                G.add_node(e1, chunks=[chunk_data], label=e1)

            # ── Thêm / cập nhật node e2 ──────────────────────
            if G.has_node(e2):
                G.nodes[e2]["chunks"].append(chunk_data)
            else:
                G.add_node(e2, chunks=[chunk_data], label=e2)

            # ── Thêm / cập nhật edge ─────────────────────────
            if G.has_edge(e1, e2):
                G[e1][e2]["weight"] += 1
                if relationship not in G[e1][e2]["rels"]:
                    G[e1][e2]["rels"].append(relationship)
            else:
                G.add_edge(e1, e2, weight=1, rels=[relationship])

    # ── Tính embedding cho từng node (batch) ─────────────────
    node_list = list(G.nodes())
    if node_list:
        try:
            node_embeddings = embedder.embed_documents(node_list)
            for node, emb in zip(node_list, node_embeddings):
                G.nodes[node]["embedding"] = emb
        except Exception:
            pass  # Nếu embedding lỗi vẫn hoạt động được (fallback keyword)

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
    Graph RAG Retrieval — 3 tầng theo đúng chuẩn Microsoft GraphRAG:

    Tầng 1 — Semantic Node Matching (embedding similarity):
        Embed câu hỏi → so sánh cosine similarity với embedding của từng node.
        Node nào similarity cao nhất → "seed nodes".
        Đây là điểm khác biệt cốt lõi so với keyword matching cũ.

    Tầng 2 — Graph Traversal (neighbor expansion):
        Từ seed nodes, duyệt sang neighbors trong graph.
        Neighbor được chọn dựa trên edge weight (số lần đồng xuất hiện trong LLM extraction).
        Mang lại thông tin liên quan gián tiếp mà pure vector search bỏ sót.

    Tầng 3 — Chunk Deduplication + Ranking:
        Gom tất cả chunks từ seed + neighbor nodes.
        Score = node_similarity × node_weight + neighbor_bonus.
        Deduplicate theo chunk_idx, giữ score cao nhất.

    Fallback: nếu không có node nào có embedding (graph cũ hoặc lỗi),
    tự động về keyword scoring để không crash.
    """
    MIN_SIM       = 0.25   # ngưỡng cosine similarity để chọn seed node
    MAX_SEED      = 5      # tối đa seed nodes
    MAX_NEIGHBORS = 3      # tối đa neighbors per seed

    node_list = list(graph.nodes())
    if not node_list:
        return []

    # ── Tầng 1: Semantic Node Matching ───────────────────────
    # Lấy embeddings của nodes (đã tính lúc build_graph_rag)
    nodes_with_emb = [
        (n, graph.nodes[n]["embedding"])
        for n in node_list
        if "embedding" in graph.nodes[n]
    ]

    seed_nodes: dict = {}   # node → similarity score

    if nodes_with_emb:
        # Embed câu hỏi
        embedder = get_embedder()
        q_emb = np.array(embedder.embed_query(question))
        q_norm = np.linalg.norm(q_emb)

        # Tính cosine similarity với tất cả nodes
        for node, node_emb in nodes_with_emb:
            n_emb = np.array(node_emb)
            n_norm = np.linalg.norm(n_emb)
            if q_norm == 0 or n_norm == 0:
                continue
            sim = float(np.dot(q_emb, n_emb) / (q_norm * n_norm))
            if sim >= MIN_SIM:
                seed_nodes[node] = sim

        # Giữ top MAX_SEED seeds
        seed_nodes = dict(
            sorted(seed_nodes.items(), key=lambda x: x[1], reverse=True)[:MAX_SEED]
        )
    else:
        # Fallback: không có embedding → dùng keyword match đơn giản
        keywords = [
            w.lower() for w in re.findall(r'[\w\u00C0-\u024F\u1E00-\u1EFF]+', question)
            if len(w) >= 4
        ]
        for node in node_list:
            nl = node.lower()
            if any(kw in nl or nl in kw for kw in keywords):
                seed_nodes[node] = 0.7

    if not seed_nodes:
        # Không tìm được seed nào → fallback lấy tất cả chunks rank bằng keyword
        keywords = [
            w.lower() for w in re.findall(r'[\w\u00C0-\u024F\u1E00-\u1EFF]+', question)
            if len(w) >= 3
        ]
        all_chunks: dict = {}
        for node in node_list:
            for ci in graph.nodes[node].get("chunks", []):
                idx = ci["chunk_idx"]
                if idx not in all_chunks:
                    s = _score_chunk(ci["content"], keywords)
                    all_chunks[idx] = (s, ci)
        ranked_fb = sorted(all_chunks.values(), key=lambda x: x[0], reverse=True)
        return [ci for _, ci in ranked_fb[:top_k] if _ > 0]

    # ── Tầng 2: Graph Traversal ────────────────────────────────
    neighbor_nodes: dict = {}   # node → neighbor_score
    for node, sim in seed_nodes.items():
        nbrs = list(graph.neighbors(node))
        if not nbrs:
            continue
        # Sort neighbors theo edge weight (LLM-extracted relation frequency)
        nbrs_sorted = sorted(
            nbrs,
            key=lambda n: graph[node][n].get("weight", 0),
            reverse=True
        )
        max_w = graph[node][nbrs_sorted[0]].get("weight", 1) if nbrs_sorted else 1
        for nb in nbrs_sorted[:MAX_NEIGHBORS]:
            if nb in seed_nodes:
                continue   # đã là seed, không cần ghi đè
            edge_w = graph[node][nb].get("weight", 1)
            # Neighbor score = seed_sim × 0.6 × (relative edge weight)
            nb_score = sim * 0.6 * (edge_w / max_w)
            if nb not in neighbor_nodes or neighbor_nodes[nb] < nb_score:
                neighbor_nodes[nb] = nb_score

    # ── Tầng 3: Gom chunks + Dedup + Rank ─────────────────────
    chunk_scores: dict = {}   # chunk_idx → (score, chunk_data)

    for node, sim in seed_nodes.items():
        if not graph.has_node(node):
            continue
        for ci in graph.nodes[node].get("chunks", []):
            idx = ci["chunk_idx"]
            final = sim   # score = cosine similarity của seed node
            if idx not in chunk_scores or chunk_scores[idx][0] < final:
                chunk_scores[idx] = (final, ci)

    for node, nb_score in neighbor_nodes.items():
        if not graph.has_node(node):
            continue
        for ci in graph.nodes[node].get("chunks", []):
            idx = ci["chunk_idx"]
            if idx not in chunk_scores or chunk_scores[idx][0] < nb_score:
                chunk_scores[idx] = (nb_score, ci)

    ranked = sorted(chunk_scores.values(), key=lambda x: x[0], reverse=True)
    return [ci for _, ci in ranked[:top_k]]


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
    lang = _detect_language(question)
    history_text = _format_chat_history(chat_history or [])
    llm = _get_llm()

    # ── Phần 1: Classic RAG 
    yield "@@CLASSIC_START@@"

    classic_docs = retriever.invoke(question)[:20]
    selected_file = CONFIG.get("selected_file", "All")
    if selected_file != "All":
        classic_docs = [d for d in classic_docs if d.metadata.get("source") == selected_file]

    if CONFIG.get("use_rerank", False):
        source_docs = rerank_documents(question, classic_docs, top_k=CONFIG["retriever_k"])
    else:
        source_docs = classic_docs[:CONFIG["retriever_k"]]

    context_classic = "\n\n".join(d.page_content for d in source_docs)
    classic_prompt = _build_prompt(lang)
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

    # ── Phần 2: Graph RAG 
    yield "@@GRAPH_START@@"

    graph_top_k = 10          
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
        docs = retriever.invoke(new_query)[:10]

        if CONFIG.get("use_rerank", False):
            docs = rerank_documents(new_query, docs)

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