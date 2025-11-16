# Đề án: Trợ lý phân loại cảm xúc tiếng Việt sử dụng Transformer

## I. Thông tin chung

- **Tên đề án**: Trợ lý phân loại cảm xúc tiếng Việt (Vietnamese Sentiment Assistant) sử dụng Transformer
- **Mục đích**: Phát triển ứng dụng phân loại cảm xúc (tích cực, trung tính, tiêu cực) từ văn bản tiếng Việt sử dụng Transformer.
- **Số lượng thành viên**: 1 – 2 sinh viên
- **Thời gian thực hiện**: 06/12/2025
- **Ngôn ngữ lập trình**: Python
- **Thư viện chính**: Hugging Face Transformers (gợi ý: `phobert-base-v2` hoặc `distilbert-base-multilingual-cased`), Underthesea (tùy chọn)
- **Giao diện**: Không giới hạn (Streamlit, Tkinter, Flask...)
- **Yêu cầu bắt buộc**: Ứng dụng chạy độc lập, phân loại cảm xúc tiếng Việt, lưu kết quả cục bộ

## II. Mục tiêu đề án

1. Xây dựng ứng dụng phân loại cảm xúc đơn giản, nhận câu tiếng Việt và trả về nhãn cảm xúc (`POSITIVE`, `NEUTRAL`, `NEGATIVE`).
2. Tích hợp Transformer pre-trained (PhoBERT hoặc DistilBERT) qua pipeline `sentiment-analysis` để phân loại, không cần fine-tuning cho bản tối giản.
3. Lưu trữ lịch sử phân loại cục bộ bằng SQLite.
4. Đảm bảo độ chính xác phân loại ≥ 65% trên 10 test case tiếng Việt.
5. Trình bày kết quả qua báo cáo đề án.

## III. Yêu cầu kỹ thuật

1. Chức năng bắt buộc

- **Nhập liệu ngôn ngữ tự nhiên**: Người dùng nhập câu tiếng Việt tự do (ví dụ: "Hôm nay tôi rất vui" hoặc "Món ăn này dở quá").
- **Phân loại cảm xúc (NLP)**: Sử dụng Transformer pre-trained để phân loại thành: `POSITIVE` (tích cực), `NEUTRAL` (trung tính), `NEGATIVE` (tiêu cực).
- **Lưu trữ cục bộ**: Lưu lịch sử phân loại (câu, nhãn cảm xúc, thời gian).
- **Hiển thị kết quả**: Hiển thị nhãn cảm xúc và danh sách lịch sử phân loại.

2. Yêu cầu về xử lý tiếng Việt

- Đầu vào: Câu tiếng Việt, có thể viết tắt hoặc thiếu dấu.
- Đầu ra: Dictionary chứa 2 trường: `text` và `sentiment`.

Yêu cầu xử lý:
- Phân loại đúng 3 nhãn: `POSITIVE`, `NEUTRAL`, `NEGATIVE`.
- Hiểu các biến thể tiếng Việt (viết tắt, thiếu dấu) ở mức cơ bản.
- Độ chính xác phân loại: ≥ 65% trên 10 test case.

3. Giao diện người dùng (tối thiểu)

- Cho phép nhập văn bản tự do.
- Nút "Phân loại cảm xúc" để gửi câu qua pipeline Transformer.
- Hiển thị nhãn cảm xúc (ví dụ: "Tích cực").
- Danh sách lịch sử phân loại (hàng hoặc list).
- Thông báo pop-up nếu nhập lỗi (ví dụ: "Câu quá ngắn").

## IV. Sản phẩm nộp (Deliverables)

1. Ứng dụng chạy được: `.exe` / Web / Python script (chạy độc lập, không lỗi)
2. Mã nguồn: Trình bày trong phần phụ lục của báo cáo đề án (có `README.md`, cấu trúc rõ ràng)
3. Báo cáo đề án: PDF theo mẫu (giới thiệu, phân tích, thiết kế, giải pháp, triển khai & kết quả, đánh giá hiệu suất, hướng dẫn cài đặt & sử dụng, kết luận)
4. Video demo: MP4 (1–2 phút), quay màn hình, có âm thanh
5. Bộ test case: 10 câu tiếng Việt + kết quả mong đợi

## V. Báo cáo đề án (cấu trúc bắt buộc gợi ý)

1. Giới thiệu & Mục tiêu
2. Phân tích yêu cầu
3. Thiết kế hệ thống (sơ đồ khối, Flowchart)
4. Giải pháp (Mô tả cách dùng Transformer)
5. Triển khai & Kết quả
6. Đánh giá hiệu suất (Bảng test 10 câu, độ chính xác)
7. Hướng dẫn cài đặt & sử dụng
8. Kết luận & Hướng phát triển

## VI. Rubrics chấm điểm (tóm tắt)

- Ứng dụng chạy ổn định & Giao diện: 3.0 điểm
- Tích hợp NLP hiệu quả (độ chính xác ≥ 65% trên 10 test): 3.0 điểm
- Xử lý ngôn ngữ tự nhiên tiếng Việt: 2.0 điểm
- Lưu trữ lịch sử phân loại: 1.5 điểm
- Báo cáo, mã nguồn, demo: 0.5 điểm

## VII. Hướng dẫn triển khai (dành cho sinh viên) — Tóm tắt kỹ thuật

Kiến trúc sử dụng Transformer pre-trained (gợi ý: `phobert-base-v2` hoặc `distilbert-base-multilingual-cased`) qua pipeline `sentiment-analysis` của Hugging Face. Không cần fine-tuning cho bản đơn giản.

Các bước chính:

1. Tiền xử lý (tuỳ chọn): chuẩn hoá câu tiếng Việt (bỏ nhiều khoảng trắng, tách từ nếu cần, sửa lỗi thường gặp).
2. Phân loại cảm xúc: Sử dụng `pipeline('sentiment-analysis', model=...)` từ Transformers.
3. Hợp nhất & xử lý lỗi: Trả về dictionary `{ "text": ..., "sentiment": ... }`, lưu vào SQLite.

Lưu ý kỹ thuật:
- Tránh SQL injection bằng parameterized queries khi chèn vào SQLite.
- Giới hạn danh sách lịch sử khi hiển thị (ví dụ: 50 bản ghi gần nhất).
- Nếu model trả xác suất < 0.5, có thể gán `NEUTRAL` như mặc định.

## VIII. Bộ Test Case (10 câu)

Danh sách 10 câu mẫu và kết quả mong đợi (theo đề án):

1. "Hôm nay tôi rất vui" → POSITIVE
2. "Món ăn này dở quá" → NEGATIVE
3. "Thời tiết bình thường" → NEUTRAL
4. "Rất vui hôm nay" → POSITIVE
5. "Công việc ổn định" → NEUTRAL
6. "Phim này hay lắm" → POSITIVE
7. "Tôi buồn vì thất bại" → NEGATIVE
8. "Ngày mai đi học" → NEUTRAL
9. "Cảm ơn rất nhiều" → POSITIVE
10. "Mệt mỏi quá hôm nay" → NEGATIVE

Ví dụ JSON đầu vào / đầu ra:

```json
{ "text": "Hôm nay tôi rất vui" }

{ "text": "Hôm nay tôi rất vui", "sentiment": "POSITIVE" }
```

## IX. Hướng dẫn cài đặt nhanh (gợi ý)

1. Tạo môi trường ảo và cài dependencies (Python 3.8+)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install transformers torch sentencepiece sqlite3 streamlit underthesea
```

Lưu ý: `sqlite3` là module tiêu chuẩn của Python, không cần cài đặt riêng; `underthesea` là tùy chọn.

2. Chạy ứng dụng (ví dụ Streamlit)

```powershell
streamlit run app.py
```

3. Hoặc chạy script Python trực tiếp

```powershell
python main.py
```

## X. Gợi ý mã nguồn tối thiểu

Đoạn ví dụ sử dụng `transformers` pipeline:

```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis', model='phobert-base-v2')

def predict(text: str):
    res = classifier(text)
    # res thường là [{'label': 'POSITIVE', 'score': 0.99}]
    label = res[0]['label']
    return { 'text': text, 'sentiment': label }
```

Gợi ý lưu vào SQLite bằng parameterized query:

```python
import sqlite3
from datetime import datetime

conn = sqlite3.connect('sentiments.db')
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS sentiments
               (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, timestamp TEXT)''')

def save_record(text, sentiment):
    ts = datetime.utcnow().isoformat(sep=' ')
    cur.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)', (text, sentiment, ts))
    conn.commit()
```

## XI. Tài liệu tham khảo

1. Hugging Face Transformers
2. VinAI PhoBERT
3. Underthesea Documentation
4. Streamlit Documentation

---

Nếu bạn muốn, tôi có thể tiếp tục và:

- Thêm một `app.py` mẫu (Streamlit) chạy được ngay.
- Thêm `requirements.txt` chính xác.
- Tạo file `sentiments.db` mẫu hoặc script khởi tạo.

Bạn muốn tôi làm tiếp phần nào?
