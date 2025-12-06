import streamlit as st
import sqlite3
from datetime import datetime
import os

# Thiết lập cache directory cho Hugging Face models
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.model_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

# Delay importing heavy ML libs until needed to avoid import-time errors in Streamlit
_TRANSFORMERS_IMPORTED = False

def _import_transformers_for_inference():
    global _TRANSFORMERS_IMPORTED
    if not _TRANSFORMERS_IMPORTED:
        # import locally to avoid ImportError during Streamlit import phase
        from transformers import pipeline
        _TRANSFORMERS_IMPORTED = True
        return pipeline
    else:
        from transformers import pipeline
        return pipeline

def add_vietnamese_accents(text):
    """Thêm dấu tiếng Việt cho văn bản thiếu dấu và mở rộng từ viết tắt"""
    # Dictionary mapping từ viết tắt -> từ đầy đủ
    abbreviation_map = {
        'k': 'không', 'K': 'Không', 
        'ko': 'không', 'Ko': 'Không',
        'kh': 'không', 'Kh': 'Không',
        'dc': 'được', 'Dc': 'Được',
        'đc': 'được', 'Đc': 'Được',
        'cx': 'cũng', 'Cx': 'Cũng',
        'j': 'gì', 'J': 'Gì',
        'gi': 'gì', 'Gi': 'Gì',
        'mk': 'mình', 'Mk': 'Mình',
        'mik': 'mình', 'Mik': 'Mình',
        'bn': 'bạn', 'Bn': 'Bạn',
        'bik': 'biết', 'Bik': 'Biết',
        'bt': 'biết', 'Bt': 'Biết',
        'ntn': 'như thế nào', 'Ntn': 'Như thế nào',
        'nt': 'nhắn tin', 'Nt': 'Nhắn tin',
        'vs': 'với', 'Vs': 'Với',
        'v': 'vậy', 'V': 'Vậy',
        'r': 'rồi', 'R': 'Rồi',
        'ny': 'này', 'Ny': 'Này',
        'oy': 'này', 'Oy': 'Này',
        'nko': 'nhỉ', 'Nko': 'Nhỉ',
        'wa': 'quá', 'Wa': 'Quá',
        'qá': 'quá', 'Qá': 'Quá',
        'nka': 'nhà', 'Nka': 'Nhà',
        'trc': 'trước', 'Trc': 'Trước',
        'bh': 'bây giờ', 'Bh': 'Bây giờ',
        'h': 'giờ', 'H': 'Giờ',
        'lm': 'làm', 'Lm': 'Làm',
        'ms': 'mới', 'Ms': 'Mới',
    }
    
    # Dictionary mapping từ không dấu -> có dấu (10-20 từ phổ biến như yêu cầu)
    accent_map = {
        'rat': 'rất', 'Rat': 'Rất',
        'vui': 'vui', 'Vui': 'Vui',
        'hom': 'hôm', 'Hom': 'Hôm',
        'nay': 'nay', 'Nay': 'Nay',
        'toi': 'tôi', 'Toi': 'Tôi',
        'buon': 'buồn', 'Buon': 'Buồn',
        'do': 'dở', 'Do': 'Dở',
        'qua': 'quá', 'Qua': 'Quá',
        'met': 'mệt', 'Met': 'Mệt',
        'moi': 'mỏi', 'Moi': 'Mỏi',
        'cam': 'cảm', 'Cam': 'Cảm',
        'on': 'ơn', 'On': 'Ơn',
        'nhieu': 'nhiều', 'Nhieu': 'Nhiều',
        'hay': 'hay', 'Hay': 'Hay',
        'lam': 'lắm', 'Lam': 'Lắm',
        'mon': 'món', 'Mon': 'Món',
        'an': 'ăn', 'An': 'Ăn',
        'thoi': 'thời', 'Thoi': 'Thời',
        'tiet': 'tiết', 'Tiet': 'Tiết',
        'binh': 'bình', 'Binh': 'Bình',
        'thuong': 'thường', 'Thuong': 'Thường',
        'cong': 'công', 'Cong': 'Công',
        'viec': 'việc', 'Viec': 'Việc',
        'dinh': 'định', 'Dinh': 'Định',
        'phim': 'phim', 'Phim': 'Phim',
        'vi': 'vì', 'Vi': 'Vì',
        'that': 'thất', 'That': 'Thất',
        'bai': 'bại', 'Bai': 'Bại',
        'ngay': 'ngày', 'Ngay': 'Ngày',
        'mai': 'mai', 'Mai': 'Mai',
        'di': 'đi', 'Di': 'Đi',
        'hoc': 'học', 'Hoc': 'Học',
        'duoc': 'được', 'Duoc': 'Được',
        'khong': 'không', 'Khong': 'Không',
        'tot': 'tốt', 'Tot': 'Tốt',
        'dep': 'đẹp', 'Dep': 'Đẹp',
        'xau': 'xấu', 'Xau': 'Xấu',
        'ban': 'bạn', 'Ban': 'Bạn',
    }
    
    words = text.split()
    result = []
    
    for word in words:
        # Bước 1: Xử lý viết tắt trước
        if word in abbreviation_map:
            result.append(abbreviation_map[word])
        # Bước 2: Thêm dấu cho từ không dấu
        elif word in accent_map:
            result.append(accent_map[word])
        else:
            result.append(word)
    
    return ' '.join(result)

DB_PATH = 'sentiments.db'

@st.cache_resource
def get_classifier(use_custom=False):  # Tạm thời tắt custom model
    pipeline_fn = _import_transformers_for_inference()
    custom_model_path = './fine_tuned_model'
    
    # Nếu có model đã fine-tune, sử dụng nó
    if use_custom and os.path.exists(custom_model_path):
        try:
            return pipeline_fn('sentiment-analysis', model=custom_model_path, local_files_only=True)
        except Exception as e:
            st.warning(f'Không thể load model đã fine-tune: {e}')
    
    # Sử dụng PhoBERT - model tốt nhất cho tiếng Việt
    preferred_models = [
        'wonrax/phobert-base-vietnamese-sentiment',  # PhoBERT fine-tuned cho sentiment tiếng Việt
        'VoVanPhuc/supernet-tiny-vietnamese-sentiment',  # Backup model tiếng Việt
        'nlptown/bert-base-multilingual-uncased-sentiment',  # Fallback multilingual
    ]
    
    for m in preferred_models:
        try:
            # Thử load từ cache trước (local_files_only=True)
            try:
                classifier = pipeline_fn('sentiment-analysis', model=m, local_files_only=True)
                st.success(f'✅ Đã load model từ cache: {m}')
                return classifier
            except Exception:
                # Nếu chưa có trong cache, download về
                st.info(f'⏬ Đang tải model lần đầu: {m} (sẽ cache cho lần sau)...')
                classifier = pipeline_fn('sentiment-analysis', model=m)
                st.success(f'✅ Đã tải và cache model: {m}')
                return classifier
        except Exception as e:
            st.warning(f'Không load được {m}: {e}, thử model khác...')
            continue
    
    # fallback to pipeline default model
    return pipeline_fn('sentiment-analysis')

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS sentiments
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, timestamp TEXT)''')
    conn.commit()
    return conn

def save_record(conn, text, sentiment):
    ts = datetime.utcnow().isoformat(sep=' ')
    cur = conn.cursor()
    cur.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)', (text, sentiment, ts))
    conn.commit()

def fetch_history(conn, limit=50):
    cur = conn.cursor()
    cur.execute('SELECT text, sentiment, timestamp FROM sentiments ORDER BY id DESC LIMIT ?', (limit,))
    return cur.fetchall()

def predict_label(classifier, text):
    try:
        # Danh sách từ khóa trung lập có confidence cao
        neutral_keywords = [
            'ổn định', 'on dinh', 'ổn', 'on',
            'bình thường', 'binh thuong', 'bình thuong', 'binh thường',
            'được đấy', 'duoc day', 'được day', 'duoc đấy',
            'công việc', 'cong viec', 'làm việc', 'lam viec',
            'thông báo', 'thong bao', 'cuộc họp', 'cuoc hop',
            'lịch', 'lich', 'ngày', 'ngay', 'thứ', 'thu',
            'đi học', 'di hoc', 'đi làm', 'di lam',
            'như mọi ngày', 'nhu moi ngay', 'hàng ngày', 'hang ngay',
            'không có gì', 'khong co gi', 'bình thường', 'thường',
        ]
        
        # Kiểm tra từ khóa trung lập
        text_lower = text.lower()
        has_neutral_keyword = any(kw in text_lower for kw in neutral_keywords)
        
        res = classifier(text)
        if isinstance(res, list) and len(res) > 0:
            label = res[0].get('label')
            score = res[0].get('score', 0.0)
            
            # Chuẩn hóa label về tiếng Việt dễ hiểu
            label_map = {
                # PhoBERT labels
                'POSITIVE': 'TÍCH CỰC',
                'NEGATIVE': 'TIÊU CỰC',
                'NEUTRAL': 'TRUNG LẬP',
                # Alternative formats
                'POS': 'TÍCH CỰC',
                'NEG': 'TIÊU CỰC',
                'NEU': 'TRUNG LẬP',
                # Generic labels
                'LABEL_0': 'TIÊU CỰC',
                'LABEL_1': 'TRUNG LẬP', 
                'LABEL_2': 'TÍCH CỰC',
                # Star ratings
                '1 star': 'RẤT TIÊU CỰC',
                '2 stars': 'TIÊU CỰC',
                '3 stars': 'TRUNG LẬP',
                '4 stars': 'TÍCH CỰC',
                '5 stars': 'RẤT TÍCH CỰC',
            }
            
            # Map tiếng Việt sang tiếng Anh cho JSON output
            vietnamese_to_english = {
                'TÍCH CỰC': 'POSITIVE',
                'TIÊU CỰC': 'NEGATIVE',
                'TRUNG LẬP': 'NEUTRAL',
                'RẤT TÍCH CỰC': 'POSITIVE',
                'RẤT TIÊU CỰC': 'NEGATIVE',
            }
            
            normalized_label = label_map.get(label, label)
            english_label = vietnamese_to_english.get(normalized_label, label)
            
            # Boost confidence cho câu trung lập có từ khóa đặc trưng
            if has_neutral_keyword and 'TRUNG LẬP' in normalized_label:
                # Tăng confidence lên tối thiểu 75% cho các từ khóa trung lập rõ ràng
                score = max(score, 0.75)
            
            return normalized_label, score, label, english_label  # Trả về cả label tiếng Anh
    except Exception as e:
        st.error(f'Error calling model: {e}')
    return 'TRUNG LẬP', 0.0, 'NEUTRAL', 'NEUTRAL'

def main():
    # Page config để tận dụng toàn bộ width
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")
    
    # Custom CSS với màu sắc nhẹ nhàng hơn và chữ lớn hơn
    st.markdown("""
    <style>
    /* Tận dụng toàn bộ width */
    .main > div {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
        font-size: 1.1rem;
    }
    
    /* Header đơn giản, màu nhẹ */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #e2e8f0;
        margin: 0.3rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Button với màu nhẹ nhàng */
    .stButton>button {
        width: 100%;
        background: #4299e1;
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.2rem;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: #3182ce;
    }
    
    /* Giảm padding của columns */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Tăng kích thước chữ cho text area */
    textarea {
        font-size: 1.1rem !important;
    }
    
    /* Tăng kích thước chữ cho markdown */
    .stMarkdown {
        font-size: 1.1rem;
    }
    
    /* Tăng kích thước chữ cho caption */
    .stCaption {
        font-size: 0.95rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    

    conn = get_conn()

    # Chia layout 60-40 thay vì 3-2
    col_left, col_right = st.columns([6, 4], gap="medium")
    
    # === CỘT TRÁI ===
    with col_left:
        st.markdown("#### 💬 Nhập văn bản")
        # Header đơn giản
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Phân loại cảm xúc tiếng Việt</h1>
        </div>
        """, unsafe_allow_html=True)
        with st.form('input_form', clear_on_submit=False):
            text = st.text_area(
                'Câu tiếng Việt',
                placeholder='Nhập câu có ít nhất 5 ký tự...',
                height=120,
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button('🚀 Phân loại', use_container_width=True)

        if submitted:
            if not text or len(text.strip()) < 5:
                st.error('⚠️ Vui lòng nhập câu có ít nhất 5 ký tự.')
            else:
                processed_text = add_vietnamese_accents(text)
                
                with st.spinner('Đang phân tích...'):
                    classifier = get_classifier()
                    label, score, original_label, english_label = predict_label(classifier, processed_text)
                    save_record(conn, processed_text, label)
                
                st.markdown("---")
                
                # Kết quả với màu nhẹ nhàng
                col_r1, col_r2, col_r3 = st.columns([3, 2, 2])
                
                with col_r1:
                    st.markdown(f"**Cảm xúc:** {label}")
                
                with col_r2:
                    if 'TÍCH CỰC' in label:
                        st.markdown("😊 Tích cực")
                    elif 'TIÊU CỰC' in label:
                        st.markdown("😞 Tiêu cực")
                    else:
                        st.markdown("😐 Trung lập")
                
                with col_r3:
                    st.markdown(f"**{score:.1%}** tin cậy")
                
                # Progress bar
                st.progress(score)
                
                # Văn bản đã xử lý (nếu có)
                if processed_text != text:
                    with st.expander("✨ Văn bản chuẩn hóa"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Trước:")
                            st.text(text)
                        with col2:
                            st.caption("Sau:")
                            st.text(processed_text)
                
                # JSON
                import json
                json_output = {
                    "text": processed_text,
                    "sentiment": english_label,
                    "confidence": round(score, 4)
                }
                with st.expander("📋 JSON"):
                    st.code(json.dumps(json_output, ensure_ascii=False, indent=2), language='json')
    
    # === CỘT PHẢI ===
    with col_right:
        st.markdown("#### 📜 Lịch sử")
        
        rows = fetch_history(conn, limit=25)
        
        if rows:
            st.caption(f"Hiển thị {len(rows)} gần nhất")
            
            for idx, (t, s, ts) in enumerate(rows, 1):
                # Màu pastel nhẹ nhàng
                if 'TÍCH CỰC' in s:
                    bg_color = "#e6f4ea"
                    border_color = "#5bb974"
                    icon = "✓"
                elif 'TIÊU CỰC' in s:
                    bg_color = "#fce8e6"
                    border_color = "#e67c73"
                    icon = "✗"
                else:
                    bg_color = "#e8f0fe"
                    border_color = "#669df6"
                    icon = "−"
                
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border-left: 3px solid {border_color};
                    border-radius: 4px;
                    padding: 12px;
                    margin: 8px 0;
                    font-size: 1rem;
                ">
                    <div style="color: #666; margin-bottom: 6px; font-size: 0.95rem;">
                        {icon} <b>#{idx}</b> • {ts[5:16]}
                    </div>
                    <div style="color: #333; margin-bottom: 6px; font-size: 1.05rem;">
                        {t[:70]}{'...' if len(t) > 70 else ''}
                    </div>
                    <div style="color: {border_color}; font-weight: 500; font-size: 1.05rem;">
                        {s}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info('Chưa có lịch sử')

if __name__ == '__main__':
    main()
