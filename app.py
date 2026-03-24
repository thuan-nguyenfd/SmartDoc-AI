import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ── Cố gắng dùng langchain-huggingface mới, fallback sang cũ ──────────────────
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ─────────────────────────────────────────────────────────────────────────────
# ĐỔI IP NÀY nếu Windows host thay đổi
# Lấy IP Windows từ WSL: cat /etc/resolv.conf | grep nameserver
# hoặc ip route | grep default
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.1.40:11434")

st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

st.title("📄 SmartDoc AI")
st.caption("Hỏi đáp tài liệu PDF · RAG + Qwen2.5:7b")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.markdown(f"""
    **Ollama Host:** `{OLLAMA_HOST}`  
    **Model:** `qwen2.5:7b`  
    **Embedding:** MPNet 768-dim  
    **Vector DB:** FAISS  
    """)

    # Kiểm tra kết nối Ollama
    import urllib.request
    try:
        urllib.request.urlopen(OLLAMA_HOST, timeout=3)
        st.success("✅ Ollama: đang chạy")
    except Exception:
        st.error("❌ Ollama: không kết nối được")
        st.info(f"Kiểm tra lại IP: `{OLLAMA_HOST}`")

    st.divider()
    st.markdown("""
    **Hướng dẫn:**
    1. Upload file PDF
    2. Chờ xử lý xong
    3. Nhập câu hỏi
    """)

    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Xóa tài liệu"):
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.rerun()

# ─── Session state ────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Load embedder (cache để không tải lại) ───────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải embedding model...")
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ─── Xử lý PDF ────────────────────────────────────────────────────────────────
def process_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embedder = load_embedder()
    vector_store = FAISS.from_documents(chunks, embedder)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return retriever, len(chunks)

# ─── Upload PDF ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Tải lên file PDF", type=["pdf"])

if uploaded_file is not None and st.session_state.retriever is None:
    with st.spinner("⏳ Đang xử lý tài liệu..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            retriever, num_chunks = process_pdf(tmp_path)
            st.session_state.retriever = retriever
            st.success(f"✅ Đã xử lý xong! {num_chunks} chunks · Sẵn sàng hỏi đáp.")
        except Exception as e:
            st.error(f"❌ Lỗi xử lý PDF: {e}")
        finally:
            os.unlink(tmp_path)

# ─── Khu vực hỏi đáp ─────────────────────────────────────────────────────────
st.divider()

if st.session_state.retriever is None:
    st.info("👆 Vui lòng upload file PDF trước.")
else:
    # Hiển thị lịch sử
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])

    # Input
    user_question = st.chat_input("Nhập câu hỏi...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang tìm câu trả lời..."):
                try:
                    # Phát hiện ngôn ngữ
                    viet_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
                    is_viet = any(c in user_question.lower() for c in viet_chars)

                    if is_viet:
                        template = """Sử dụng ngữ cảnh sau để trả lời câu hỏi.
Nếu không biết, hãy nói không biết. Trả lời ngắn gọn bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời:"""
                    else:
                        template = """Use the context below to answer the question.
If you don't know, say so. Keep the answer concise.

Context: {context}

Question: {question}

Answer:"""

                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["context", "question"]
                    )

                    # Trỏ đúng OLLAMA_HOST
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

                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })

                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")