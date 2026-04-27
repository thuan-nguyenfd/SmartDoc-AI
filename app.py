import os
import tempfile
import urllib.request
import uuid
import datetime
import json
import streamlit as st

from database import (
    init_db, save_message, load_history,
    load_all_sessions, delete_session, clear_all_history,
)
from rag_engine import (
    get_embedder, process_pdf, process_docx,
    ask_question_stream_with_sources,
    ask_compare_rag, build_graph_rag,
)
from rag_engine_graph_optimized import streamlit_build_graph_with_progress
from styles import APP_CSS

# ══════════════════════════════════════════════════════════════
# KHỞI TẠO
# ══════════════════════════════════════════════════════════════

init_db()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

_defaults = {
    "retriever": None,
    "graph": None,            # Graph RAG graph object
    "chunks_store": None,     # lưu chunks để build graph khi cần
    "chat_history": [],
    "pdf_info": None,
    "view_session": None,
    "ollama_ok": None,
    "sessions_cache": None,
    "sessions_dirty": True,
    "uploader_key": 0,
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "top_k": 5,
    "documents": [],
    "selected_file": "All",
    "use_rerank": False,
    "rag_mode": "Basic RAG",
    "compare_mode": False,    # True = chế độ so sánh Classic vs Graph
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if not st.session_state.chat_history:
    rows = load_history(st.session_state.session_id)
    st.session_state.chat_history = [
        {"question": q, "answer": a, "sources": s, "timestamp": t}
        for q, a, s, t in rows
    ]

# ══════════════════════════════════════════════════════════════
# CACHE / HELPERS
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Đang tải embedding model...")
def _cached_embedder():
    return get_embedder()


def _get_ollama_status() -> bool:
    if st.session_state.ollama_ok is None:
        try:
            urllib.request.urlopen(OLLAMA_HOST, timeout=2)
            st.session_state.ollama_ok = True
        except Exception:
            st.session_state.ollama_ok = False
    return st.session_state.ollama_ok


def _get_sessions():
    if st.session_state.sessions_dirty or st.session_state.sessions_cache is None:
        st.session_state.sessions_cache = load_all_sessions()
        st.session_state.sessions_dirty = False
    return st.session_state.sessions_cache


def _mark_sessions_dirty():
    st.session_state.sessions_dirty = True


# ══════════════════════════════════════════════════════════════
# DIALOGS
# ══════════════════════════════════════════════════════════════

@st.dialog("Xóa tất cả lịch sử?")
def _dialog_clear_all_history():
    st.warning("**Toàn bộ** lịch sử của mọi phiên sẽ bị xóa vĩnh viễn.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Xác nhận", type="primary", use_container_width=True):
            clear_all_history()
            st.session_state.chat_history = []
            st.session_state.view_session = None
            _mark_sessions_dirty()
            st.rerun()
    with c2:
        if st.button("Hủy", use_container_width=True):
            st.rerun()


@st.dialog("Xóa tài liệu đang tải?")
def _dialog_delete_pdf():
    info = st.session_state.pdf_info
    name = info.get("names", ["tài liệu hiện tại"])[0] if info else "tài liệu hiện tại"
    st.warning(f"Tài liệu **{name}** sẽ bị gỡ khỏi hệ thống.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Xác nhận", type="primary", use_container_width=True):
            st.session_state.uploader_key += 1
            st.session_state.retriever = None
            st.session_state.graph = None
            st.session_state.chunks_store = None
            st.session_state.pdf_info = None
            st.session_state.documents = []
            st.rerun()
    with c2:
        if st.button("Hủy", use_container_width=True):
            st.rerun()


# ══════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════

def _highlight_text(content: str, question: str, answer: str = "") -> str:
    """
    Highlight đoạn văn là NGUỒN THỰC SỰ của câu trả lời.

    THUẬT TOÁN: N-gram Source Tracing
    ─────────────────────────────────────────────────────────
    Ý tưởng cốt lõi: LLM paraphrase nội dung từ chunk → câu trong answer
    và câu trong chunk sẽ chia sẻ nhiều CỤM TỪ (bigram/trigram) giống nhau,
    dù không giống từng từ.

    Vì sao tốt hơn bag-of-words:
    - BOW: "Hồ Chí Minh" xuất hiện ở MỌI câu → mọi câu đều score cao
    - N-gram: "gia đình nhà nho yêu nước" chỉ xuất hiện ở đúng đoạn nguồn

    Pipeline:
    1. Tách câu trong chunk thành danh sách câu hoàn chỉnh (≥ 40 ký tự)
    2. Tách câu trong answer thành danh sách câu
    3. Với mỗi câu trong chunk, tính Jaccard similarity của bigrams với
       từng câu trong answer → lấy max
    4. Chỉ highlight nếu score ≥ ngưỡng động (tránh false positive)
    5. Span-based rendering (không dùng set lookup)
    ─────────────────────────────────────────────────────────
    """
    import re

    def escape(text: str) -> str:
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

    # ── Bước 1: Tách câu trong chunk ─────────────────────────────────────────
    # Chuẩn hóa: single newline → space (PDF wrap), giữ paragraph break
    normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)
    normalized = re.sub(r' {2,}', ' ', normalized)

    # Tách theo dấu câu tiếng Việt, paragraph break
    # Lookbehind: sau . ! ? …  — Lookahead: chữ hoa hoặc số hoặc bullet
    SENT_SPLIT = re.compile(
        r'(?<=[.!?…])\s+'
        r'(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ0-9\-])'
        r'|\n\n+'
    )
    raw_sents = SENT_SPLIT.split(normalized)
    # Chỉ giữ câu đủ dài — lọc header/số trang/fragment
    chunk_sents = [s.strip() for s in raw_sents if len(s.strip()) >= 40]

    if not chunk_sents:
        return escape(content)

    # ── Bước 2: Tách câu trong answer ────────────────────────────────────────
    answer_sents = []
    if answer and len(answer.strip()) >= 20:
        ans_normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', answer)
        ans_raw = SENT_SPLIT.split(ans_normalized)
        answer_sents = [s.strip() for s in ans_raw if len(s.strip()) >= 20]

    # Nếu không có answer, fallback về question
    if not answer_sents and question:
        answer_sents = [question]

    if not answer_sents:
        return escape(content)

    # ── Bước 3: Bigram Jaccard similarity ────────────────────────────────────
    STOP_WORDS = {
        "là", "của", "và", "các", "có", "cho", "trong", "với", "về", "được",
        "này", "đó", "khi", "nào", "như", "hay", "hoặc", "tôi", "bạn",
        "hãy", "theo", "đã", "một", "những", "từ", "sau", "tại", "đến",
        "không", "bởi", "vì", "nên", "mà", "lên", "ra", "vào", "đi",
        "the", "of", "and", "for", "in", "is", "to", "a", "an",
        "it", "on", "at", "by", "or", "be", "that", "this", "are", "was",
    }

    def tokenize(text: str) -> list:
        tokens = re.findall(r'[\w\u00C0-\u024F\u1E00-\u1EFF]+', text.lower())
        return [t for t in tokens if len(t) >= 2 and t not in STOP_WORDS]

    def get_bigrams(tokens: list) -> set:
        if len(tokens) < 2:
            return set(tokens)  # fallback: unigrams nếu quá ngắn
        return {(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)}

    def get_trigrams(tokens: list) -> set:
        if len(tokens) < 3:
            return set()
        return {(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens) - 2)}

    def jaccard(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0

    def score_chunk_sent(chunk_sent: str) -> float:
        """
        Score = max Jaccard(bigrams_chunk_sent, bigrams_answer_sent)
        Dùng cả bigram + trigram, trigram có trọng số cao hơn vì đặc trưng hơn.
        """
        c_tokens = tokenize(chunk_sent)
        c_bigrams = get_bigrams(c_tokens)
        c_trigrams = get_trigrams(c_tokens)

        best = 0.0
        for a_sent in answer_sents:
            a_tokens = tokenize(a_sent)
            a_bigrams = get_bigrams(a_tokens)
            a_trigrams = get_trigrams(a_tokens)

            bi_score = jaccard(c_bigrams, a_bigrams)
            tri_score = jaccard(c_trigrams, a_trigrams) if c_trigrams and a_trigrams else 0.0

            # Trigram đặc trưng hơn → weight cao hơn
            combined = 0.4 * bi_score + 0.6 * tri_score if tri_score > 0 else bi_score
            best = max(best, combined)
        return best

    scored_sents = [(score_chunk_sent(s), s) for s in chunk_sents]
    scored_sents.sort(key=lambda x: x[0], reverse=True)

    # ── Bước 4: Ngưỡng động ──────────────────────────────────────────────────
    if not scored_sents:
        return escape(content)

    max_score = scored_sents[0][0]

    # Không highlight gì nếu score tốt nhất quá thấp
    # (nghĩa là không có câu nào thực sự là nguồn của answer)
    ABS_MIN = 0.05          # ngưỡng tuyệt đối tối thiểu
    REL_MIN_RATIO = 0.5     # câu được chọn phải ≥ 50% score của câu tốt nhất

    if max_score < ABS_MIN:
        return escape(content)

    dynamic_threshold = max(ABS_MIN, max_score * REL_MIN_RATIO)

    highlight_sents = []
    for sc, sent in scored_sents:
        if sc < dynamic_threshold:
            break
        if len(highlight_sents) >= 2:   # tối đa 2 câu highlight / chunk
            break
        highlight_sents.append(sent)

    if not highlight_sents:
        return escape(content)

    # ── Bước 5: Span-based rendering ─────────────────────────────────────────
    # Tìm vị trí (start, end) của từng câu cần highlight trong content GỐC
    exact_spans = []
    for sent in highlight_sents:
        # Thử match 25 ký tự đầu của câu trong content gốc
        probe_len = min(25, len(sent))
        probe = sent[:probe_len].strip()
        # Bỏ ký tự đặc biệt regex trong probe
        probe_escaped = re.escape(probe)
        # Tìm trong content gốc
        m = re.search(probe_escaped, content)
        if not m:
            # Thử probe ngắn hơn (15 ký tự)
            probe2 = re.escape(sent[:15].strip())
            m = re.search(probe2, content)
        if not m:
            continue

        start = m.start()
        # Ước tính end: tìm dấu câu kết thúc sau start
        search_end = content[start: start + int(len(sent) * 1.5) + 50]
        end_m = re.search(r'[.!?…](?:\s|$)', search_end)
        if end_m:
            end = start + end_m.end()
        else:
            end = start + len(sent)
        end = min(end, len(content))

        if end > start:
            exact_spans.append((start, end))

    if not exact_spans:
        return escape(content)

    # Sắp xếp và merge spans liền kề
    exact_spans.sort()
    merged = [list(exact_spans[0])]
    for s, e in exact_spans[1:]:
        if s <= merged[-1][1] + 10:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Render HTML
    result_parts = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            result_parts.append(escape(content[cursor:start]))
        result_parts.append(
            f"<mark style='background:linear-gradient(120deg,#FFD700 0%,#FFF176 100%);"
            f"padding:2px 5px;border-radius:4px;font-weight:600;"
            f"display:inline'>{escape(content[start:end])}</mark>"
        )
        cursor = end

    if cursor < len(content):
        result_parts.append(escape(content[cursor:]))

    return "".join(result_parts)


def _render_sources(sources_data: list, question: str = "", answer: str = "", key_prefix: str = ""):
    if not sources_data:
        return
    with st.expander(f"📄 Nguồn tham khảo ({len(sources_data)} đoạn)"):
        for src in sources_data:
            page_display = src['page'] + 1 if isinstance(src['page'], int) else src['page']
            file_name = src.get('source', '')
            if file_name and ('\\' in file_name or '/' in file_name):
                file_name = file_name.replace('\\\\', '/').replace('\\', '/').split('/')[-1]
            st.markdown(f"**Đoạn {src['index']} — Trang {page_display}** · 📄 `{file_name}`")
            highlighted = _highlight_text(src['content'], question, answer) if (question or answer) else \
                src['content'].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f"<div style='background:#f8f9fa;border-left:3px solid #007BFF;"
                f"padding:10px 14px;border-radius:4px;font-size:13px;"
                f"line-height:1.6;white-space:pre-wrap'>{highlighted}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## SMARTDOC AI")
    st.caption("RAG · Qwen2.5 · FAISS · Graph")
    st.divider()

    st.markdown("### Hướng dẫn")
    st.markdown("""
1. Upload file PDF/DOCX  
2. Chờ hệ thống xử lý  
3. Nhập câu hỏi  
4. Nhận câu trả lời  
    """)
    st.divider()

    st.markdown("### Cài đặt")

    # ── Chế độ RAG ──────────────────────────────────────────
    compare_mode = st.toggle(
        "So sánh RAG vs Graph RAG",
        value=st.session_state.compare_mode,
        help="Bật để xem câu trả lời song song từ 2 phương pháp RAG khác nhau. Chỉ hỗ trợ 1 file."
    )
    st.session_state.compare_mode = compare_mode

    if not compare_mode:
        use_rerank = st.checkbox(
            "Bật Re-ranking (Cross-Encoder)",
            value=st.session_state.use_rerank
        )
        st.session_state.use_rerank = use_rerank

        rag_mode = st.radio(
            "Chọn RAG:",
            ["Basic RAG", "Self-RAG"],
            index=0 if st.session_state.rag_mode == "Basic RAG" else 1,
            horizontal=True,
        )
        st.session_state.rag_mode = rag_mode
    else:
        st.info("Chế độ so sánh: Classic RAG ↔ Graph RAG.")
        st.session_state.rag_mode = "Basic RAG"

    # Sync CONFIG
    from rag_engine import CONFIG
    CONFIG["use_rerank"] = st.session_state.use_rerank
    CONFIG["use_self_rag"] = (st.session_state.rag_mode == "Self-RAG")

    # ── Filter tài liệu (chỉ hiện khi không compare) ────────
    selected_file = "All"
    if st.session_state.pdf_info and not compare_mode:
        selected_file = st.selectbox(
            "Filter theo tài liệu",
            ["All"] + st.session_state.pdf_info["names"]
        )
    st.session_state.selected_file = selected_file
    CONFIG["selected_file"] = selected_file

    # ── Chunk / Top K ────────────────────────────────────────
    chunk_size = st.selectbox(
        "Chunk Size",
        [500, 1000, 1500, 2000],
        index=[500, 1000, 1500, 2000].index(st.session_state.chunk_size)
    )
    chunk_overlap = st.selectbox(
        "Chunk Overlap",
        [50, 100, 200, 300],
        index=[50, 100, 200, 300].index(st.session_state.chunk_overlap)
    )
    top_k = st.selectbox(
        "Top K",
        [3, 5, 7, 10],
        index=[3, 5, 7, 10].index(st.session_state.top_k)
    )

    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    st.session_state.top_k = top_k
    CONFIG["retriever_k"] = top_k

    if st.session_state.pdf_info:
        prev_cs = st.session_state.pdf_info.get("chunk_size")
        prev_co = st.session_state.pdf_info.get("chunk_overlap")
        if prev_cs is not None and (prev_cs != chunk_size or prev_co != chunk_overlap):
            st.warning("⚠️ Chunk params đã thay đổi. Upload lại tài liệu để áp dụng.")

    st.markdown(f"""
        <div class="setting-item"><span>Filter</span><code>{selected_file}</code></div>
        <div class="setting-item"><span>Chunk Size</span><code>{chunk_size}</code></div>
        <div class="setting-item"><span>Chunk Overlap</span><code>{chunk_overlap}</code></div>
        <div class="setting-item"><span>Top K</span><code>{top_k}</code></div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Model")
    st.markdown(f"""
        <div class="setting-item"><span>LLM</span><code>{CONFIG['llm_model']}</code></div>
        <div class="setting-item"><span>Embedding</span><code>mpnet-base-v2</code></div>
        <div class="setting-item"><span>Vector DB</span><code>FAISS</code></div>
        <div class="setting-item"><span>Framework</span><code>LangChain</code></div>
    """, unsafe_allow_html=True)

    st.divider()

    if _get_ollama_status():
        st.success("🟢 Ollama đang chạy")
    else:
        st.error("🔴 Ollama chưa kết nối")
        st.caption(f"Host: `{OLLAMA_HOST}`")
        if st.button("Kiểm tra lại"):
            st.session_state.ollama_ok = None
            st.rerun()

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = []
            st.session_state.retriever = None
            st.session_state.graph = None
            st.session_state.chunks_store = None
            st.session_state.pdf_info = None
            st.session_state.view_session = None
            st.session_state.uploader_key += 1
            st.session_state.documents = []
            _mark_sessions_dirty()
            st.rerun()
    with col2:
        if st.button("🗑 Xóa tài liệu", use_container_width=True,
                     disabled=st.session_state.pdf_info is None):
            _dialog_delete_pdf()

    st.divider()

    st.markdown("### Lịch sử hội thoại")
    all_sessions = _get_sessions()
    if not all_sessions:
        st.caption("Chưa có lịch sử nào.")
    else:
        if st.button("Xóa tất cả lịch sử", use_container_width=True):
            _dialog_clear_all_history()
        st.markdown("")
        for sid, pdf_name, cnt, started in all_sessions:
            is_current = (sid == st.session_state.session_id)
            badge = " ●" if is_current else ""
            label = f"📄 {pdf_name or 'Unknown'}{badge}"
            sub = f"{cnt} câu · {started[:10]}"
            if st.button(f"{label} | {sub}", key=f"sess_{sid}", use_container_width=True):
                st.session_state.view_session = sid
                st.rerun()

    st.caption("Spring 2026")

# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-title">
    Hỏi Đáp Tài Liệu PDF <span class="rag-badge">RAG</span>
</div>
<div class="subtitle">
    Tải lên tài liệu PDF và trò chuyện với AI để nhận thông tin chính xác, nhanh chóng
</div>
""", unsafe_allow_html=True)

# ── Upload file ───────────────────────────────────────────────
_accept_multiple = not st.session_state.compare_mode

uploaded_files = st.file_uploader(
    "Kéo thả hoặc click để tải lên file (PDF hoặc DOCX - tối đa 20MB)"
    + (" · Chế độ so sánh: chỉ 1 file" if st.session_state.compare_mode else ""),
    type=["pdf", "docx"],
    key=f"uploader_{st.session_state.uploader_key}",
    accept_multiple_files=_accept_multiple,
)

# Chuẩn hóa thành list
if uploaded_files is None:
    uploaded_files = []
elif not isinstance(uploaded_files, list):
    uploaded_files = [uploaded_files]

# Giới hạn 1 file khi compare mode
if st.session_state.compare_mode and len(uploaded_files) > 1:
    st.warning("Chế độ so sánh chỉ hỗ trợ 1 file. Chỉ file đầu tiên sẽ được xử lý.")
    uploaded_files = [uploaded_files[0]]

if uploaded_files:
    for file in uploaded_files:
        if file.size > 20 * 1024 * 1024:
            st.error("File quá lớn! Vui lòng chọn file dưới 20MB.")
            st.stop()

if not uploaded_files and st.session_state.retriever is not None:
    st.session_state.retriever = None
    st.session_state.graph = None
    st.session_state.chunks_store = None
    st.session_state.documents = []

current_files = st.session_state.pdf_info.get("names", []) if st.session_state.pdf_info else []
new_files = [f.name for f in uploaded_files] if uploaded_files else []
need_process = uploaded_files and (
    st.session_state.retriever is None or set(new_files) != set(current_files)
)

if need_process:
    file_names = ", ".join([f.name for f in uploaded_files])
    with st.spinner(f"Đang xử lý {len(uploaded_files)} file: {file_names} ..."):
        all_chunks = []
        file_metadata = []
        embedder = _cached_embedder()

        for file in uploaded_files:
            ext = file.name.lower().split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            try:
                with st.spinner(f"→ Đang xử lý {file.name}..."):
                    if ext == "pdf":
                        chunks, pages, num_chunks = process_pdf(
                            tmp_path, embedder,
                            chunk_size=st.session_state.chunk_size,
                            chunk_overlap=st.session_state.chunk_overlap,
                        )
                    elif ext == "docx":
                        chunks, pages, num_chunks = process_docx(
                            tmp_path, embedder,
                            chunk_size=st.session_state.chunk_size,
                            chunk_overlap=st.session_state.chunk_overlap,
                        )
                    else:
                        continue

                    for chunk in chunks:
                        chunk.metadata["source"] = file.name
                        chunk.metadata["file_type"] = ext
                        chunk.metadata["upload_time"] = str(datetime.datetime.now())

                    all_chunks.extend(chunks)
                    del chunks
                    file_metadata.append({"name": file.name, "pages": pages, "chunks": num_chunks})
            finally:
                os.unlink(tmp_path)

        from rag_engine import build_hybrid_retriever
        retriever = build_hybrid_retriever(all_chunks, embedder)
        st.session_state.retriever = retriever
        st.session_state.chunks_store = all_chunks  # lưu lại để build graph

        # Build Graph RAG nếu đang ở compare mode
        if st.session_state.compare_mode:
            st.session_state.graph = streamlit_build_graph_with_progress(all_chunks)
            node_count = st.session_state.graph.number_of_nodes()
            edge_count = st.session_state.graph.number_of_edges()
            st.success(f"Knowledge Graph: {node_count} nodes · {edge_count} edges")

        del all_chunks

        st.session_state.pdf_info = {
            "names": [f["name"] for f in file_metadata],
            "files": file_metadata,
            "chunk_size": st.session_state.chunk_size,
            "chunk_overlap": st.session_state.chunk_overlap,
        }
        st.session_state.documents = file_metadata
        _mark_sessions_dirty()

# Khi bật compare mode sau khi đã upload file → build graph nếu chưa có
if (st.session_state.compare_mode
        and st.session_state.retriever is not None
        and st.session_state.graph is None
        and st.session_state.chunks_store is not None):
    with st.spinner("Đang xây dựng Knowledge Graph từ tài liệu hiện tại..."):
        st.session_state.graph = streamlit_build_graph_with_progress(st.session_state.chunks_store)
        node_count = st.session_state.graph.number_of_nodes()
        edge_count = st.session_state.graph.number_of_edges()
        st.success(f"Knowledge Graph: {node_count} nodes · {edge_count} edges")

# Hiển thị thông tin tài liệu
if st.session_state.pdf_info and st.session_state.documents:
    st.markdown("### 📂 Tài liệu đã tải")
    for doc in st.session_state.documents:
        file_icon = "📄" if doc["name"].lower().endswith(".pdf") else "📝"
        st.markdown(f"""
        <div class="status-row">
            <span class="status-chip">{file_icon} {doc['name']}</span>
            <span class="status-chip">📑 {doc['pages']} trang</span>
            <span class="status-chip">🧩 {doc['chunks']} chunks</span>
        </div>
        """, unsafe_allow_html=True)

    # Hiển thị Graph info nếu đang compare mode
    if st.session_state.compare_mode and st.session_state.graph is not None:
        G = st.session_state.graph
        st.markdown(f"""
        <div class="status-row">
            <span class="status-chip ready">🕸️ Graph RAG sẵn sàng</span>
            <span class="status-chip">{G.number_of_nodes()} nodes</span>
            <span class="status-chip">{G.number_of_edges()} edges</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# CHAT AREA
# ══════════════════════════════════════════════════════════════

if st.session_state.view_session and st.session_state.view_session != st.session_state.session_id:
    vs = st.session_state.view_session
    hist = load_history(vs)
    if hist:
        st.info(f"Đang xem lịch sử session · {len(hist)} câu hỏi")
        if st.button("← Quay lại session hiện tại"):
            st.session_state.view_session = None
            st.rerun()
        for q, a, src_json, ts in hist:
            st.markdown(f"<small style='color:#6c757d'>🕐 {ts}</small>", unsafe_allow_html=True)
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
                sources = json.loads(src_json) if src_json else []
                _render_sources(sources, question=q, answer=a)
    else:
        st.warning("Không tìm thấy lịch sử cho session này.")
        st.session_state.view_session = None

else:
    # Hiển thị lịch sử chat hiện tại
    for item in st.session_state.chat_history:
        ts = item.get("timestamp", "")
        if ts:
            st.markdown(f"<small style='color:#6c757d'>🕐 {ts}</small>", unsafe_allow_html=True)
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            # Kiểm tra xem có phải compare result không
            if item.get("is_compare"):
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("#### RAG")
                    st.markdown(item.get("classic_answer", ""))
                    _render_sources(
                        json.loads(item.get("classic_sources", "[]")),
                        question=item["question"],
                        answer=item.get("classic_answer", ""),
                        key_prefix=f"hist_classic_{item.get('timestamp','')}",
                    )
                with col_r:
                    st.markdown("#### Graph RAG")
                    st.markdown(item.get("graph_answer", ""))
                    _render_sources(
                        json.loads(item.get("graph_sources", "[]")),
                        question=item["question"],
                        answer=item.get("graph_answer", ""),
                        key_prefix=f"hist_graph_{item.get('timestamp','')}",
                    )
            else:
                st.write(item["answer"])
                sources = json.loads(item["sources"]) if item.get("sources") else []
                _render_sources(sources, question=item["question"], answer=item.get("answer", ""))

    # Chat input
    if st.session_state.retriever is None:
        st.info("Upload file PDF hoặc DOCX để bắt đầu hỏi đáp.")
    else:
        user_question = st.chat_input("Nhập câu hỏi của bạn về tài liệu...")

        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

            # ══════════════════════════════════════════════════
            # COMPARE MODE: Classic RAG vs Graph RAG song song
            # ══════════════════════════════════════════════════
            if st.session_state.compare_mode:
                if st.session_state.graph is None:
                    st.warning("Graph chưa được xây dựng. Vui lòng upload lại file.")
                else:
                    col_l, col_r = st.columns(2)

                    with col_l:
                        st.markdown("#### RAG")
                        classic_placeholder = st.empty()
                        classic_placeholder.markdown(" Đang tìm câu trả lời...")
                        classic_sources_placeholder = st.empty()

                    with col_r:
                        st.markdown("#### Graph RAG")
                        graph_placeholder = st.empty()
                        graph_placeholder.markdown(" Đang xử lý sau RAG...")
                        graph_sources_placeholder = st.empty()

                    classic_answer = ""
                    classic_sources = []
                    graph_answer = ""
                    graph_sources = []
                    phase = "classic"  # classic | graph

                    try:
                        for token in ask_compare_rag(
                            user_question,
                            st.session_state.retriever,
                            st.session_state.graph,
                            chat_history=st.session_state.chat_history,
                        ):
                            if token == "@@CLASSIC_START@@":
                                phase = "classic"
                                classic_placeholder.markdown(" Đang tìm câu trả lời...")
                                continue

                            if token.startswith("@@CLASSIC_SOURCES@@"):
                                classic_sources = json.loads(token[len("@@CLASSIC_SOURCES@@"):])
                                classic_placeholder.markdown(classic_answer)
                                # Render sources trong col_l
                                with col_l:
                                    _render_sources(classic_sources, question=user_question,
                                                    answer=classic_answer, key_prefix="compare_classic")
                                continue

                            if token == "@@GRAPH_START@@":
                                phase = "graph"
                                graph_placeholder.markdown(" Đang tìm câu trả lời...")
                                continue

                            if token.startswith("@@GRAPH_SOURCES@@"):
                                graph_sources = json.loads(token[len("@@GRAPH_SOURCES@@"):])
                                graph_placeholder.markdown(graph_answer)
                                with col_r:
                                    _render_sources(graph_sources, question=user_question,
                                                    answer=graph_answer, key_prefix="compare_graph")
                                continue

                            if token == "@@COMPARE_DONE@@":
                                break

                            # Token bình thường
                            if phase == "classic":
                                classic_answer += token
                                classic_placeholder.markdown(classic_answer + "▌")
                            elif phase == "graph":
                                graph_answer += token
                                graph_placeholder.markdown(graph_answer + "▌")  # streaming Graph RAG

                        # Lưu vào DB và session
                        combined_answer = (
                            f"**Classic RAG:**\n{classic_answer}\n\n"
                            f"**Graph RAG:**\n{graph_answer}"
                        )
                        pdf_name = ", ".join(st.session_state.pdf_info["names"]) if st.session_state.pdf_info else None
                        save_message(
                            st.session_state.session_id, pdf_name,
                            user_question, combined_answer,
                            sources=classic_sources,
                        )
                        ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": combined_answer,
                            "sources": json.dumps(classic_sources, ensure_ascii=False),
                            "timestamp": ts_now,
                            "is_compare": True,
                            "classic_answer": classic_answer,
                            "graph_answer": graph_answer,
                            "classic_sources": json.dumps(classic_sources, ensure_ascii=False),
                            "graph_sources": json.dumps(graph_sources, ensure_ascii=False),
                        })
                        _mark_sessions_dirty()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Lỗi khi gọi model: {e}")
                        st.caption("Kiểm tra Ollama đang chạy và model đã được pull chưa.")

            # ══════════════════════════════════════════════════
            # NORMAL MODE: Classic / Self-RAG
            # ══════════════════════════════════════════════════
            else:
                full_answer = ""
                sources_data = []
                with st.chat_message("assistant"):
                    answer_placeholder = st.empty()
                    answer_placeholder.markdown(" Đang tìm câu trả lời...")
                    try:
                        for chunk in ask_question_stream_with_sources(
                            user_question,
                            st.session_state.retriever,
                            chat_history=st.session_state.chat_history,
                        ):
                            if chunk.startswith("@@CONFIDENCE@@"):
                                score = float(chunk.replace("@@CONFIDENCE@@", ""))
                                st.caption(f" Confidence: {score:.2f}")
                                continue
                            if chunk.startswith("@@SOURCES@@"):
                                sources_data = json.loads(chunk[len("@@SOURCES@@"):])
                                continue
                            full_answer += chunk
                            answer_placeholder.markdown(full_answer + "▌")

                        answer_placeholder.markdown(full_answer)
                        _render_sources(sources_data, question=user_question, answer=full_answer)

                        pdf_name = ", ".join(st.session_state.pdf_info["names"]) if st.session_state.pdf_info else None
                        save_message(st.session_state.session_id, pdf_name,
                                     user_question, full_answer, sources=sources_data)
                        ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": full_answer,
                            "sources": json.dumps(sources_data, ensure_ascii=False) if sources_data else None,
                            "timestamp": ts_now,
                        })
                        _mark_sessions_dirty()
                        st.rerun()

                    except Exception as e:
                        answer_placeholder.error(f"Lỗi khi gọi model: {e}")
                        st.caption("Kiểm tra Ollama đang chạy và model đã được pull chưa.")