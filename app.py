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
            
            normalized_label = label_map.get(label, label)
            return normalized_label, score, label  # Trả về cả label gốc
    except Exception as e:
        st.error(f'Error calling model: {e}')
    return 'TRUNG LẬP', 0.0, 'NEUTRAL'

def main():
    st.title('Trợ lý phân loại cảm xúc tiếng Việt (Transformer)')
    st.write('Nhập câu tiếng Việt và nhấn `Phân loại cảm xúc`.')

    conn = get_conn()

    with st.form('input_form'):
        text = st.text_input('Câu tiếng Việt', '')
        submitted = st.form_submit_button('Phân loại cảm xúc')

    if submitted:
        if not text or len(text.strip()) < 2:
            st.warning('Vui lòng nhập câu có độ dài hợp lệ (>=2 ký tự).')
        else:
            with st.spinner('Tải model và phân loại (lần đầu có thể chậm)...'):
                classifier = get_classifier()
                label, score, original_label = predict_label(classifier, text)
                save_record(conn, text, label)
            st.success(f'Kết quả: {label} (score={score:.2f})')
            st.caption(f'Label gốc từ model: {original_label}')

    st.subheader('Lịch sử phân loại (gần nhất)')
    rows = fetch_history(conn, limit=50)
    if rows:
        for t, s, ts in rows:
            st.write(f'[{ts}] {t} → **{s}**')
    else:
        st.info('Chưa có bản ghi nào.')

if __name__ == '__main__':
    main()
