import streamlit as st
import tempfile
import os
import subprocess

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

import sqlite3
import datetime

# ══════════════════════════════════════════════════════════════
# SQLite – Lưu trữ lịch sử hội thoại
# ══════════════════════════════════════════════════════════════
DB_PATH = "chat_history.db"

def init_db():
    """Khởi tạo database và bảng lịch sử."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session   TEXT NOT NULL,
            pdf_name  TEXT,
            question  TEXT NOT NULL,
            answer    TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_message(session_id: str, pdf_name: str, question: str, answer: str):
    """Lưu một cặp Q&A vào SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (session, pdf_name, question, answer, timestamp) VALUES (?,?,?,?,?)",
        (session_id, pdf_name, question, answer,
         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def load_history(session_id: str):
    """Tải lịch sử của session hiện tại."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT question, answer, timestamp FROM conversations WHERE session=? ORDER BY id",
        (session_id,)
    )
    rows = c.fetchall()
    conn.close()
    return rows  # [(question, answer, timestamp), ...]

def load_all_sessions():
    """Tải danh sách các session đã có (để hiển thị sidebar)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT session, pdf_name, COUNT(*) as cnt,
               MIN(timestamp) as started
        FROM conversations
        GROUP BY session
        ORDER BY started DESC
        LIMIT 20
    """)
    rows = c.fetchall()
    conn.close()
    return rows  # [(session_id, pdf_name, count, started), ...]

def delete_session(session_id: str):
    """Xóa toàn bộ lịch sử của một session."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE session=?", (session_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    """Xóa toàn bộ lịch sử."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()

# Khởi tạo DB ngay khi app chạy
init_db()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');

/* ── Toàn trang ── */
html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif !important;
}

/* ── Background chính ── */
.stApp {
    background-color: #F8F9FA;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #2C2F33 !important;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.80) !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: rgba(255,255,255,0.08) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    width: 100%;
    transition: background .2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* ── Header title ── */
.main-title {
    font-size: 26px;
    font-weight: 700;
    color: #212529;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.rag-badge {
    background: #007BFF;
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 9px;
    border-radius: 5px;
}
.subtitle {
    color: #6c757d;
    font-size: 13.5px;
    margin-bottom: 20px;
}

/* ── Upload box ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #dee2e6 !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #007BFF !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Upload button (amber) ── */
[data-testid="stFileUploaderDropzoneInput"] + div button,
[data-testid="stBaseButton-secondary"] {
    background-color: #FFC107 !important;
    color: #212529 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: background .2s;
}
[data-testid="stBaseButton-secondary"]:hover {
    background-color: #e0a800 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}

/* User message – highlight xanh nhạt */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #e8f0fe !important;
    border-color: #b3cdf9 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border: 1.5px solid #dee2e6 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #007BFF !important;
    box-shadow: 0 0 0 3px rgba(0,123,255,0.12) !important;
}

/* ── Send button (primary blue) ── */
[data-testid="stChatInputSubmitButton"] {
    background-color: #007BFF !important;
    border-radius: 8px !important;
    color: #fff !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    background-color: #0056c7 !important;
}

/* ── Success / Error / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 13px !important;
}

/* ── Status metrics ── */
.status-row {
    display: flex;
    gap: 12px;
    margin: 10px 0 6px;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #212529;
    font-weight: 500;
}
.status-chip.ready {
    border-color: #28a745;
    color: #28a745;
}

/* ── Divider ── */
hr {
    border-color: #dee2e6 !important;
    margin: 16px 0 !important;
}

/* ── Sidebar setting item ── */
.setting-item {
    display: flex;
    justify-content: space-between;
    padding: 5px 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    margin-bottom: 5px;
    font-size: 12px;
}
.setting-item code {
    color: #FFC107;
    font-family: monospace;
    font-size: 11px;
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
import uuid
if "retriever"     not in st.session_state: st.session_state.retriever     = None
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "pdf_info"      not in st.session_state: st.session_state.pdf_info      = None
if "session_id"    not in st.session_state: st.session_state.session_id    = str(uuid.uuid4())[:8]
if "view_session"  not in st.session_state: st.session_state.view_session  = None

# Khôi phục chat_history từ SQLite khi reload
if not st.session_state.chat_history:
    rows = load_history(st.session_state.session_id)
    st.session_state.chat_history = [
        {"question": q, "answer": a, "timestamp": t} for q, a, t in rows
    ]

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 SmartDoc AI")
    st.caption("RAG · Qwen2.5 · FAISS")
    st.divider()

    # Hướng dẫn
    st.markdown("### Hướng dẫn")
    st.markdown("""
1. Upload file PDF  
2. Chờ hệ thống xử lý  
3. Nhập câu hỏi  
4. Nhận câu trả lời  
    """)
    st.divider()

    # Cài đặt
    st.markdown("### Cài đặt")
    st.markdown("""
<div class="setting-item"><span>Chunk Size</span><code>1000</code></div>
<div class="setting-item"><span>Chunk Overlap</span><code>100</code></div>
<div class="setting-item"><span>Top K</span><code>3</code></div>
<div class="setting-item"><span>Temperature</span><code>0.7</code></div>
<div class="setting-item"><span>Device</span><code>CPU</code></div>
    """, unsafe_allow_html=True)
    st.divider()

    # Model config
    st.markdown("### Model")
    st.markdown("""
<div class="setting-item"><span>LLM</span><code>qwen2.5:7b</code></div>
<div class="setting-item"><span>Embedding</span><code>mpnet-base-v2</code></div>
<div class="setting-item"><span>Vector DB</span><code>FAISS</code></div>
<div class="setting-item"><span>Framework</span><code>LangChain</code></div>
    """, unsafe_allow_html=True)
    st.divider()

    # Kiểm tra Ollama
    import urllib.request
    try:
        urllib.request.urlopen(OLLAMA_HOST, timeout=3)
        st.success("🟢 Ollama đang chạy")
    except Exception:
        st.error("🔴 Ollama chưa kết nối")
        st.caption(f"Host: `{OLLAMA_HOST}`")
    st.divider()

    # Nút xóa
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Xóa chat"):
            delete_session(st.session_state.session_id)
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Xóa PDF"):
            st.session_state.retriever    = None
            st.session_state.chat_history = []
            st.session_state.pdf_info     = None
            st.rerun()

    st.divider()

    # ── Lịch sử hội thoại (SQLite) ────────────────────────────
    st.markdown("### Lịch sử hội thoại")

    all_sessions = load_all_sessions()

    if not all_sessions:
        st.caption("Chưa có lịch sử nào.")
    else:
        if st.button("Xóa tất cả lịch sử", use_container_width=True):
            clear_all_history()
            st.session_state.chat_history = []
            st.session_state.view_session = None
            st.rerun()

        st.markdown("")
        for sid, pdf_name, cnt, started in all_sessions:
            label = f"📄 {pdf_name or 'Unknown'}"
            sub   = f"{cnt} câu · {started[:10]}"
            is_current = (sid == st.session_state.session_id)
            badge = " 🟢" if is_current else ""
            btn_label = f"{label}{badge} | {sub}"
            if st.button(btn_label, key=f"sess_{sid}", use_container_width=True):
                st.session_state.view_session = sid
                st.rerun()

    st.caption("v1.2 · Spring 2026")

# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-title">
    Hỏi Đáp Tài Liệu PDF <span class="rag-badge">RAG</span>
</div>
<div class="subtitle">
    Tải lên tài liệu PDF và trò chuyện với AI để nhận thông tin chính xác, nhanh chóng
</div>
""", unsafe_allow_html=True)

# ── Cache embedding model ─────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải embedding model...")
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ── Xử lý PDF ────────────────────────────────────────────────
def process_pdf(file_path: str):
    loader = PDFPlumberLoader(file_path)
    docs   = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks   = splitter.split_documents(docs)
    embedder = load_embedder()
    vector_store = FAISS.from_documents(chunks, embedder)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return retriever, len(docs), len(chunks)

# ── Upload PDF ────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Kéo thả hoặc click để tải lên file PDF (tối đa 20MB)",
    type=["pdf"],
    help="Chỉ hỗ trợ file PDF"
)

# Validate size
if uploaded_file is not None:
    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("❌ File quá lớn! Vui lòng chọn file dưới 20MB.")
        uploaded_file = None

if uploaded_file is not None and st.session_state.retriever is None:
    with st.spinner("⏳ Đang xử lý tài liệu... vui lòng chờ"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            retriever, num_pages, num_chunks = process_pdf(tmp_path)
            st.session_state.retriever = retriever
            st.session_state.pdf_info  = {
                "name":   uploaded_file.name,
                "pages":  num_pages,
                "chunks": num_chunks
            }
        except Exception as e:
            st.error(f"❌ Lỗi xử lý PDF: {e}")
        finally:
            os.unlink(tmp_path)

# Status chips sau khi upload
if st.session_state.pdf_info:
    info = st.session_state.pdf_info
    st.markdown(f"""
    <div class="status-row">
        <span class="status-chip">📄 {info['name']}</span>
        <span class="status-chip">📑 {info['pages']} trang</span>
        <span class="status-chip">🧩 {info['chunks']} chunks</span>
        <span class="status-chip ready">✅ FAISS Ready</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Chat area ─────────────────────────────────────────────────

# Chế độ xem lịch sử session cũ
if st.session_state.view_session and st.session_state.view_session != st.session_state.session_id:
    vs = st.session_state.view_session
    hist = load_history(vs)
    if hist:
        pdf_label = hist[0][0] if hist else ""
        st.info(f"📖 Đang xem lịch sử session:  · {len(hist)} câu hỏi")
        if st.button("← Quay lại session hiện tại"):
            st.session_state.view_session = None
            st.rerun()
        for q, a, ts in hist:
            st.markdown(f"<small style='color:#6c757d'>🕐 {ts}</small>", unsafe_allow_html=True)
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
    else:
        st.warning("Không tìm thấy lịch sử cho session này.")
        st.session_state.view_session = None

elif st.session_state.retriever is None:
    st.info("💡 Vui lòng upload file PDF để bắt đầu hỏi đáp.")
else:
    # Hiện lại toàn bộ lịch sử session hiện tại
    for item in st.session_state.chat_history:
        ts = item.get("timestamp", "")
        if ts:
            st.markdown(f"<small style='color:#6c757d'>🕐 {ts}</small>", unsafe_allow_html=True)
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])

    # Input câu hỏi
    user_question = st.chat_input("Nhập câu hỏi của bạn về tài liệu...")

    if user_question:
        # Hiện tin nhắn user ngay
        with st.chat_message("user"):
            st.write(user_question)

        # Sinh câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang tìm câu trả lời..."):
                try:
                    # Phát hiện ngôn ngữ
                    viet_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
                    is_viet    = any(c in user_question.lower() for c in viet_chars)

                    template = (
                        """Sử dụng ngữ cảnh sau để trả lời câu hỏi.
Nếu không biết, hãy nói không biết. Trả lời ngắn gọn bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời:"""
                        if is_viet else
                        """Use the context below to answer the question.
If you don't know, say so. Keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
                    )

                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["context", "question"]
                    )
                    llm = Ollama(
                        model="qwen2.5:7b",
                        base_url=OLLAMA_HOST,
                        temperature=0.7
                    )
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.retriever,
                        chain_type_kwargs={"prompt": prompt}
                    )

                    result = qa_chain.invoke({"query": user_question})
                    answer = result["result"]
                    st.write(answer)

                    # Lưu vào SQLite và session_state
                    pdf_name = st.session_state.pdf_info["name"] if st.session_state.pdf_info else None
                    save_message(st.session_state.session_id, pdf_name, user_question, answer)
                    ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history.append({
                        "question":  user_question,
                        "answer":    answer,
                        "timestamp": ts_now,
                    })

                except Exception as e:
                    st.error(f"❌ Lỗi khi gọi model: {e}")
                    st.caption("Kiểm tra Ollama đang chạy và model đã được pull chưa.")