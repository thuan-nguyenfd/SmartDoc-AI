# Tất cả SQLite logic (câu 2, 3)
# ══════════════════════════════════════════════════════════════
# database.py
# Chịu trách nhiệm: Tất cả thao tác SQLite (lưu, đọc, xóa lịch sử)
#
# Để mở rộng (Câu 2, 3 trong mục 8):
#   - Thêm cột mới vào bảng conversations ở hàm init_db()
#   - Thêm hàm mới vào file này, rồi import vào app.py
# ══════════════════════════════════════════════════════════════

import sqlite3
import datetime

DB_PATH = "chat_history.db"


def init_db():
    """
    Khởi tạo database và tạo bảng conversations nếu chưa có.
    Gọi hàm này 1 lần duy nhất khi app khởi động (ở app.py).
    """
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
    """
    Lưu một cặp Q&A vào SQLite.

    Tham số:
        session_id : ID phiên làm việc hiện tại
        pdf_name   : Tên file PDF đang được hỏi
        question   : Câu hỏi của người dùng
        answer     : Câu trả lời từ LLM
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (session, pdf_name, question, answer, timestamp) VALUES (?,?,?,?,?)",
        (
            session_id,
            pdf_name,
            question,
            answer,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def load_history(session_id: str) -> list[tuple]:
    """
    Tải toàn bộ lịch sử Q&A của một session.

    Trả về:
        list of (question, answer, timestamp)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT question, answer, timestamp FROM conversations WHERE session=? ORDER BY id",
        (session_id,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def load_all_sessions() -> list[tuple]:
    """
    Tải danh sách tất cả session (dùng để hiển thị sidebar).

    Trả về:
        list of (session_id, first_question, count, started)
        - first_question: câu hỏi đầu tiên của session (thay vì tên file)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT
            session,
            MIN(question) AS first_question,
            COUNT(*)      AS cnt,
            MIN(timestamp) AS started
        FROM conversations
        GROUP BY session
        ORDER BY started DESC
        LIMIT 20
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def delete_session(session_id: str):
    """Xóa toàn bộ lịch sử Q&A của một session cụ thể."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE session=?", (session_id,))
    conn.commit()
    conn.close()


def clear_all_history():
    """Xóa toàn bộ lịch sử của mọi session."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()