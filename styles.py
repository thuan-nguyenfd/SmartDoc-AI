# CSS string
# ══════════════════════════════════════════════════════════════
# styles.py
# Chịu trách nhiệm: Toàn bộ CSS của ứng dụng
#
# Cách dùng trong app.py:
#     from styles import APP_CSS
#     st.markdown(APP_CSS, unsafe_allow_html=True)
#
# Muốn đổi màu / font → chỉ sửa file này, không đụng app.py.
# ══════════════════════════════════════════════════════════════

APP_CSS = """
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
"""