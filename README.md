# 📄 SmartDoc AI — Intelligent Document Q&A System

> Hệ thống hỏi đáp thông minh từ tài liệu PDF sử dụng RAG (Retrieval-Augmented Generation)  
> Môn học: Open Source Software Development · Spring 2026 · Trường ĐH Sài Gòn

---

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt Ollama](#1-cài-đặt-ollama-windows)
- [Pull project từ Git](#2-pull-project-từ-git)
- [Tạo môi trường ảo](#3-tạo-môi-trường-ảo)
- [Cài thư viện](#4-cài-đặt-thư-viện)
- [Cấu hình](#5-cấu-hình)
- [Chạy dự án](#6-chạy-dự-án)
- [Cách sử dụng](#cách-sử-dụng)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)

---

## Giới thiệu

SmartDoc AI cho phép người dùng **tải lên tài liệu PDF** và **đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh**. Hệ thống tự động tìm kiếm thông tin liên quan trong tài liệu và trả lời thông minh nhờ kết hợp:

| Thành phần                     | Công nghệ                  |
| ------------------------------ | -------------------------- |
| Giao diện web                  | Streamlit                  |
| Đọc PDF                        | PDFPlumber                 |
| Embedding (vector hóa văn bản) | MPNet Multilingual 768-dim |
| Vector Database                | FAISS                      |
| Mô hình ngôn ngữ               | Qwen2.5:7b (qua Ollama)    |
| Framework AI                   | LangChain                  |

---

## Yêu cầu hệ thống

| Thành phần        | Tối thiểu                               |
| ----------------- | --------------------------------------- |
| RAM               | 8 GB                                    |
| VRAM (GPU)        | 4 GB (hoặc chạy CPU)                    |
| Dung lượng ổ cứng | 10 GB (model ~4.4 GB + thư viện)        |
| Python            | 3.10 trở lên                            |
| Hệ điều hành      | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

> **Lưu ý WSL:** Nếu chạy app trong WSL nhưng Ollama trên Windows, xem phần [Cấu hình](#5-cấu-hình).

---

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

## 5. Cấu hình

### Trường hợp thông thường (Ollama và app cùng máy)

Không cần cấu hình thêm. Mặc định app kết nối `http://localhost:11434`.

### Trường hợp WSL + Ollama trên Windows

App chạy trong WSL không thể dùng `localhost` để kết nối Windows. Cần trỏ đúng IP:

**Cách 1 — Dùng biến môi trường (khuyên dùng):**

```bash
streamlit run app.py
```

## 6. Chạy dự án

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

Mở trình duyệt và truy cập:

```
http://localhost:8501
```

---

## Cách sử dụng

### Upload tài liệu

1. Nhấn **Browse files** hoặc kéo thả file PDF vào ô upload
2. Chờ thông báo ✅ — hệ thống đang tách văn bản và tạo vector embeddings
3. Quá trình này mất khoảng **10–30 giây** tùy kích thước file

### Đặt câu hỏi

1. Nhập câu hỏi vào ô chat bên dưới
2. Nhấn **Enter** hoặc nút **Gửi**
3. Chờ khoảng **5–15 giây** để model tạo câu trả lời

### Mẹo để câu trả lời chính xác hơn

- Đặt câu hỏi **cụ thể**, tránh quá chung chung
- Dùng **từ khóa** xuất hiện trong tài liệu
- Chia câu hỏi phức tạp thành **nhiều câu nhỏ**
- Hỏi bằng **tiếng Việt** → hệ thống tự động trả lời tiếng Việt

### Xóa dữ liệu

- **Xóa lịch sử chat:** Nhấn nút 🗑️ trong sidebar
- **Đổi tài liệu mới:** Nhấn nút 🔄 trong sidebar rồi upload lại

---

## Cấu trúc thư mục

```
SmartDoc-AI/
│
├── app.py                  ← File chính — chạy lệnh streamlit run app.py
├── requirements.txt        ← Danh sách thư viện Python cần cài
├── README.md               ← File này
│
├── data/                   ← Thư mục chứa PDF mẫu để test
│   └── sample.pdf
│
└── venv/                   ← Môi trường ảo Python (không commit lên Git)
```

> **Lưu ý:** Thư mục `venv/` đã được thêm vào `.gitignore`, không cần commit lên Git.

## Tài liệu tham khảo

- [LangChain Documentation](https://python.langchain.com)
- [Ollama](https://ollama.ai)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://docs.streamlit.io)
- [Sentence Transformers](https://www.sbert.net)
