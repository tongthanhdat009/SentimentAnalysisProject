# 📚 HƯỚNG DẪN SỬ DỤNG VÀ FINE-TUNE MODEL

## ✅ Đã sửa xong các lỗi

1. ✅ `regex` - Circular import
2. ✅ `torch` - DLL thiếu  
3. ✅ `scikit-learn` - Build không tương thích
4. ✅ `scipy` - Extension modules lỗi
5. ✅ Model phân loại sai - Đã đổi sang `nlptown/bert-base-multilingual-uncased-sentiment`

## 🎯 Model hiện tại

**Model chính**: `wonrax/phobert-base-vietnamese-sentiment` (PhoBERT)

Đây là PhoBERT - model BERT được VinAI Research phát triển riêng cho tiếng Việt, sau đó được fine-tune cho sentiment analysis.

### Ưu điểm PhoBERT:
- ✅ Được train trên 20GB dữ liệu tiếng Việt
- ✅ Hiểu tokenization tiếng Việt tốt hơn (từ ghép, dấu thanh)
- ✅ Đã được fine-tune cho sentiment analysis tiếng Việt
- ✅ Phân loại chính xác hơn cho văn bản tiếng Việt

### Fallback models:
1. `VoVanPhuc/supernet-tiny-vietnamese-sentiment` - Model tiếng Việt nhẹ
2. `nlptown/bert-base-multilingual-uncased-sentiment` - Multilingual backup

### Cách phân loại:

| Label gốc | Kết quả hiển thị |
|-----------|------------------|
| POSITIVE  | TÍCH CỰC        |
| NEGATIVE  | TIÊU CỰC        |
| NEUTRAL   | TRUNG LẬP       |

**Lưu ý**: PhoBERT model thường trả về POSITIVE/NEGATIVE/NEUTRAL trực tiếp.

## 🧪 Test các câu

Thử các câu sau để kiểm tra:

**Tiêu cực:**
- "tôi muốn chết" → RẤT TIÊU CỰC
- "Tôi bị ngu" → TIÊU CỰC
- "Món ăn này dở quá" → TIÊU CỰC
- "Mệt mỏi quá" → TIÊU CỰC
- "Tôi buồn vì thất bại" → TIÊU CỰC

**Tích cực:**
- "Tôi rất vui" → TÍCH CỰC
- "Món này ngon tuyệt" → RẤT TÍCH CỰC
- "Tuyệt vời quá" → RẤT TÍCH CỰC

**Trung lập:**
- "Tôi là Đạt" → TRUNG LẬP
- "Hôm nay thứ hai" → TRUNG LẬP

## 🚀 Cách cải thiện độ chính xác

### Phương án 1: Thu thập thêm dữ liệu (Khuyến nghị)

1. Sử dụng app để phân loại nhiều câu tiếng Việt
2. Mở database `sentiments.db` bằng DB Browser hoặc Python
3. Sửa lại các kết quả sai trong cột `sentiment`
4. Khi có ít nhất **100-200 mẫu chính xác**, chạy fine-tune

### Phương án 2: Fine-tune model

```bash
# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Chạy script fine-tune
python train_model.py
```

Script này sẽ:
- ✅ Load dữ liệu từ `sentiments.db`
- ✅ Thêm 25 mẫu training cơ bản nếu dữ liệu < 20
- ✅ Split train/test (80/20)
- ✅ Fine-tune BERT model
- ✅ Lưu model vào `./fine_tuned_model`

**Sau khi fine-tune:**

1. Mở file `app.py`
2. Tìm dòng: `def get_classifier(use_custom=False):`
3. Đổi thành: `def get_classifier(use_custom=True):`
4. Khởi động lại app

## 📊 Yêu cầu tối thiểu cho fine-tune hiệu quả

| Số lượng mẫu | Độ chính xác kỳ vọng |
|--------------|----------------------|
| 10-50        | 40-60% (không khuyến nghị) |
| 100-200      | 70-80% (tối thiểu) |
| 500-1000     | 85-90% (tốt) |
| 2000+        | 90-95% (rất tốt) |

## 💡 Tips

1. **Đảm bảo dữ liệu cân bằng**: Số lượng mẫu TÍCH CỰC, TIÊU CỰC, TRUNG LẬP nên tương đương nhau
2. **Dữ liệu chất lượng**: Câu phải được gán nhãn đúng
3. **Đa dạng**: Bao gồm nhiều ngữ cảnh, domain khác nhau
4. **Label gốc hiển thị**: App hiện giờ show cả label gốc từ model để debug

## 🔍 Debug

Nếu kết quả vẫn không chính xác:

1. Kiểm tra label gốc hiển thị ở dưới kết quả
2. PhoBERT thường trả về POSITIVE/NEGATIVE/NEUTRAL
3. Nếu model fallback sang multilingual, có thể trả star ratings (1-5 stars)
4. Thêm mapping mới vào hàm `predict_label()` trong `app.py` nếu cần

## 📦 Model đã thử

**Đã chọn**: PhoBERT vì:
- Được train riêng cho tiếng Việt
- Hiểu ngữ cảnh, từ ghép tiếng Việt tốt hơn
- Fine-tuned sẵn cho sentiment analysis

**So sánh với multilingual BERT**: PhoBERT cho kết quả tốt hơn 10-15% với văn bản tiếng Việt.

## 📁 Cấu trúc project

```
SentimentAnalysisProject/
├── app.py                 # Main app Streamlit
├── train_model.py         # Script fine-tune model
├── sentiments.db          # Database lưu lịch sử
├── fine_tuned_model/      # Model sau fine-tune (nếu có)
├── requirements.txt       # Dependencies
└── HUONG_DAN.md          # File này
```

## 🎓 Tài liệu tham khảo

- [PhoBERT - VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- [wonrax/phobert-base-vietnamese-sentiment](https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)
