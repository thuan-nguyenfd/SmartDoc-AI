# ══════════════════════════════════════════════════════════════
# app.py  —  File điều phối chính (entry point)  [v1.3 - optimized]
#
# Các tối ưu hiệu năng so với v1.2:
#   1. Ollama health-check chỉ chạy 1 lần lúc khởi động, không
#      chạy lại mỗi lần Streamlit rerun → loại bỏ 3s timeout lag
#   2. load_all_sessions() được cache vào session_state, chỉ
#      reload khi thực sự có thay đổi (xóa / thêm session mới)
#   3. LLM streaming: câu trả lời hiện ra từng chữ ngay lập tức
#      thay vì chờ model sinh xong toàn bộ rồi mới hiện
#   4. st.rerun() được thay bằng cập nhật session_state trực tiếp
#      ở những chỗ không cần full rerun (xóa chat, xóa PDF)
# ══════════════════════════════════════════════════════════════

import os
import tempfile
import urllib.request
import uuid
import datetime
import math
import streamlit as st

from database import (
    init_db,
    save_message,
    load_history,
    load_all_sessions,
    delete_session,
    clear_all_history,
)
from rag_engine import get_embedder, process_pdf, process_docx, ask_question_stream
from styles import APP_CSS

# ══════════════════════════════════════════════════════════════
# KHỞI TẠO (chỉ chạy 1 lần khi server start)
# ══════════════════════════════════════════════════════════════

init_db()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE — khởi tạo tất cả keys ở 1 chỗ
# ══════════════════════════════════════════════════════════════

_defaults = {
    "retriever":        None,
    "chat_history":     [],
    "pdf_info":         None,
    "view_session":     None,
    "ollama_ok":        None,   # None = chưa check, True/False = kết quả
    "sessions_cache":   None,   # cache load_all_sessions()
    "sessions_dirty":   True,   # True = cần reload từ DB
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Tạo session_id ổn định (chỉ tạo 1 lần)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Khôi phục lịch sử từ SQLite khi reload trang (chỉ khi list đang rỗng)
if not st.session_state.chat_history:
    rows = load_history(st.session_state.session_id)
    st.session_state.chat_history = [
        {"question": q, "answer": a, "timestamp": t} for q, a, t in rows
    ]

# ══════════════════════════════════════════════════════════════
# CACHE FUNCTIONS
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Đang tải embedding model...")
def _cached_embedder():
    """Embedding model — tải 1 lần duy nhất suốt vòng đời server."""
    return get_embedder()


def _get_ollama_status() -> bool:
    """
    FIX LAG #1: Kiểm tra Ollama chỉ 1 lần rồi cache vào session_state.
    Không gọi network mỗi lần Streamlit rerun nữa.
    """
    if st.session_state.ollama_ok is None:
        try:
            urllib.request.urlopen(OLLAMA_HOST, timeout=2)
            st.session_state.ollama_ok = True
        except Exception:
            st.session_state.ollama_ok = False
    return st.session_state.ollama_ok


def _get_sessions():
    """
    FIX LAG #2: Cache danh sách sessions vào session_state.
    Chỉ query SQLite khi sessions_dirty = True (sau khi xóa / thêm mới).
    """
    if st.session_state.sessions_dirty or st.session_state.sessions_cache is None:
        st.session_state.sessions_cache = load_all_sessions()
        st.session_state.sessions_dirty = False
    return st.session_state.sessions_cache


def _mark_sessions_dirty():
    """Gọi hàm này sau mọi thao tác thay đổi DB để invalidate cache."""
    st.session_state.sessions_dirty = True


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📄 SmartDoc AI")
    st.caption("RAG · Qwen2.5 · FAISS")
    st.divider()

    st.markdown("### Hướng dẫn")
    st.markdown("""
1. Upload file PDF  
2. Chờ hệ thống xử lý  
3. Nhập câu hỏi  
4. Nhận câu trả lời  
    """)
    st.divider()

    from rag_engine import CONFIG as RAG_CONFIG
    st.markdown("### Cài đặt")
    st.markdown(f"""
<div class="setting-item"><span>Chunk Size</span><code>{RAG_CONFIG['chunk_size']}</code></div>
<div class="setting-item"><span>Chunk Overlap</span><code>{RAG_CONFIG['chunk_overlap']}</code></div>
<div class="setting-item"><span>Top K</span><code>{RAG_CONFIG['retriever_k']}</code></div>
<div class="setting-item"><span>Temperature</span><code>{RAG_CONFIG['llm_temperature']}</code></div>
<div class="setting-item"><span>Device</span><code>{RAG_CONFIG['embedding_device'].upper()}</code></div>
    """, unsafe_allow_html=True)
    st.divider()

    # ====================== DIALOG XÁC NHẬN XÓA ======================

    @st.dialog("⚠️ Xác nhận xóa")
    def delete_confirm_dialog():
            st.caption("Hành động này không thể hoàn tác.")

            col1, col2 = st.columns(2, gap="small")

            with col1:
                if st.button("✅ Đồng ý", type="primary", use_container_width=True):
                    delete_session(st.session_state.session_id)
                    st.session_state.chat_history = []
                    _mark_sessions_dirty()
                    st.success("Đã xóa thành công!")
                    
                    st.rerun()

            with col2:
                if st.button("❌ Hủy", type="secondary", use_container_width=True):
            
                    st.rerun()


        # Nút "Xóa thử" ở Sidebar - SỬA KEY
    if st.button("Xóa thử", key="btn_delete_test_sidebar", use_container_width=True):
            delete_confirm_dialog()   



    st.markdown("### Model")
    st.markdown(f"""
<div class="setting-item"><span>LLM</span><code>{RAG_CONFIG['llm_model']}</code></div>
<div class="setting-item"><span>Embedding</span><code>mpnet-base-v2</code></div>
<div class="setting-item"><span>Vector DB</span><code>FAISS</code></div>
<div class="setting-item"><span>Framework</span><code>LangChain</code></div>
    """, unsafe_allow_html=True)

    st.divider()

    # FIX LAG #1 — dùng cached status, không gọi network mỗi rerun
    if _get_ollama_status():
        st.success("🟢 Ollama đang chạy")
    else:
        st.error("🔴 Ollama chưa kết nối")
        st.caption(f"Host: `{OLLAMA_HOST}`")
        if st.button("🔄 Kiểm tra lại"):
            st.session_state.ollama_ok = None
            st.rerun()
    st.divider()

    # Nút xóa
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Xóa chat", use_container_width=True):
            delete_confirm_dialog()
            
            # delete_session(st.session_state.session_id)
            # st.session_state.chat_history = []
            # mark_sessions_dirty()
            # st.rerun()
            
    with col2:
        if st.button("📄 Xóa PDF", use_container_width=True):
            delete_confirm_dialog_xoa_upload()
            # st.session_state.retriever    = None
            # st.session_state.chat_history = []
            # st.session_state.pdf_info     = None
            # st.rerun()
    st.divider()

    # FIX LAG #2 — dùng cached sessions
    st.markdown("### Lịch sử hội thoại")
    all_sessions = _get_sessions()

    if not all_sessions:
        st.caption("Chưa có lịch sử nào.")
    else:
        if st.button("🗑 Xóa tất cả lịch sử", use_container_width=True):
            clear_all_history()
            st.session_state.chat_history = []
            st.session_state.view_session = None
            _mark_sessions_dirty()
            st.rerun()

        st.markdown("")
        for sid, pdf_name, cnt, started in all_sessions:
            is_current = (sid == st.session_state.session_id)
            badge      = " 🟢" if is_current else ""
            label      = f"📄 {pdf_name or 'Unknown'}{badge}"
            sub        = f"{cnt} câu · {started[:10]}"
            if st.button(f"{label} | {sub}", key=f"sess_{sid}", use_container_width=True):
                st.session_state.view_session = sid
                st.rerun()

    st.caption("v1.3 · Spring 2026")

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


 



# ── Upload PDF ────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📁 Kéo thả hoặc click để tải lên file (PDF hoặc DOCX - tối đa 20MB)",
    type=["pdf", "docx"], #them xu ly file doc o day
    help="Chỉ hỗ trợ file PDF và .docx",
)

if uploaded_file is not None and uploaded_file.size > 20 * 1024 * 1024:
    st.error("❌ File quá lớn! Vui lòng chọn file dưới 20MB.")
    uploaded_file = None


#xu ly file khi ng dung upload va chua co retriever
if uploaded_file is not None and st.session_state.retriever is None:
    file_extension = uploaded_file.name.lower().split('.')[-1] # lấy đuôi file
    with st.spinner(f"⏳ Đang xử lý tài liệu { file_extension.upper() }... vui lòng chờ"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            embedder  = _cached_embedder()
            if file_extension == "pdf":
                retriever, num_pages, num_chunks = process_pdf(tmp_path, embedder)
                file_type= "PDF"
            elif file_extension == "docx":
                retriever, num_pages, num_chunks = process_docx(tmp_path, embedder)
                file_type= "DOCX"
            else:
                st.error(f"Hệ thống hiện chưa hỗ trợ định dạng file {file_extension.upper()}")
                retriever=None

            if retriever:       
                st.session_state.retriever = retriever
                st.session_state.pdf_info  = {
                    "name":   uploaded_file.name,
                    "type": file_type,
                    "pages":  num_pages,
                    "chunks": num_chunks,
                }
                _mark_sessions_dirty()
        except Exception as e:
            st.error(f"❌ Lỗi xử lý file {file_extension.upper()}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if st.session_state.pdf_info:
    info = st.session_state.pdf_info
    file_icon = "📄" if info.get("type") == "PDF" else "📝"
    st.markdown(f"""
    <div class="status-row">
        <span class="status-chip">{file_icon} {info['name']}</span>
        <span class="status-chip">📑 {info['pages']} trang</span>
        <span class="status-chip">🧩 {info['chunks']} chunks</span>
        <span class="status-chip ready">✅ FAISS Ready</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Chat area ─────────────────────────────────────────────────

if st.session_state.view_session and st.session_state.view_session != st.session_state.session_id:
    vs   = st.session_state.view_session
    hist = load_history(vs)
    if hist:
        st.info(f"📖 Đang xem lịch sử session · {len(hist)} câu hỏi")
        if st.button("← Quay lại session hiện tại"):
            st.session_state.view_session = None
            st.rerun()
        for q, a, ts in hist:
            st.markdown(f"<small style='color:#6c757d'>🕐 {ts}</small>", unsafe_allow_html=True)
            with st.chat_message("user"):      st.write(q)
            with st.chat_message("assistant"): st.write(a)
    else:
        st.warning("Không tìm thấy lịch sử cho session này.")
        st.session_state.view_session = None

elif st.session_state.retriever is None:
    st.info("💡 Vui lòng upload file PDF hoặc DOCX để bắt đầu hỏi đáp.")

else:
    for item in st.session_state.chat_history:
        ts = item.get("timestamp", "")
        if ts:
            st.markdown(f"<small style='color:#6c757d'>🕐 {ts}</small>", unsafe_allow_html=True)
        with st.chat_message("user"):      st.write(item["question"])
        with st.chat_message("assistant"): st.write(item["answer"])

    user_question = st.chat_input("Nhập câu hỏi của bạn về tài liệu...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        # FIX LAG #3 — streaming: chữ hiện ra từng token, không chờ model xong
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for chunk in ask_question_stream(user_question, st.session_state.retriever):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")   # con trỏ nhấp nháy

                answer_placeholder.markdown(full_answer)             # xóa con trỏ khi xong

                pdf_name = st.session_state.pdf_info["name"] if st.session_state.pdf_info else None
                save_message(st.session_state.session_id, pdf_name, user_question, full_answer)
                ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append({
                    "question":  user_question,
                    "answer":    full_answer,
                    "timestamp": ts_now,
                })
                _mark_sessions_dirty()

            except Exception as e:
                answer_placeholder.error(f"❌ Lỗi khi gọi model: {e}")
                st.caption("Kiểm tra Ollama đang chạy và model đã được pull chưa.")