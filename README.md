# 📄 SmartDoc AI — Intelligent Document Q&A System

## 1. Cài đặt Ollama (Windows)

Ollama là runtime để chạy LLM local. Cần cài trên **Windows**.

**Bước 1:** Tải Ollama tại https://ollama.ai → chọn **Download for Windows**

**Bước 2:** Cài đặt và kiểm tra:

```powershell
ollama --version
```

**Bước 3:** Khởi động Ollama server:

```powershell
ollama serve
```

## 2. Pull project từ Git

Mở terminal, chạy:

```bash
git https://github.com/thuan-nguyenfd/SmartDoc-AI.git
cd SmartDoc-AI
```

## 3. Tạo môi trường ảo

Tạo và kích hoạt virtual environment để tránh xung đột thư viện:

```bash
# Tạo môi trường ảo
python -m venv venv
```

```bash
# Kích hoạt — Windows PowerShell
venv\Scripts\activate
```

Sau khi kích hoạt, terminal sẽ hiển thị `(venv)` ở đầu dòng.

---

## 4. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 5. Chạy dự án

**Terminal 1 — Khởi động Ollama** (Windows PowerShell):

```powershell
ollama serve
```

**Terminal 2 — Chạy Streamlit** (WSL hoặc terminal dự án):

```bash
# Kích hoạt môi trường ảo trước
source venv/bin/activate

# Chạy app
streamlit run app.py
```

làm câu 7 trước, sau đó 9, 6, 10,8
