"""
Script để fine-tune model sentiment analysis với dữ liệu từ database
"""
import sqlite3
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

DB_PATH = 'sentiments.db'
MODEL_NAME = 'wonrax/phobert-base-vietnamese-sentiment'  # PhoBERT cho tiếng Việt
OUTPUT_DIR = './fine_tuned_model'

def load_training_data():
    """Load và chuẩn bị dữ liệu từ database"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT text, sentiment FROM sentiments"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Đã load {len(df)} mẫu từ database")
    
    # Mapping sentiment labels sang numeric
    label_map = {
        'TÍCH CỰC': 2,
        'RẤT TÍCH CỰC': 2,
        'TIÊU CỰC': 0,
        'RẤT TIÊU CỰC': 0,
        'TRUNG LẬP': 1,
    }
    
    df['label'] = df['sentiment'].map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print("\nPhân bố nhãn:")
    print(df['sentiment'].value_counts())
    
    return df

def prepare_dataset(df):
    """Chuẩn bị dataset cho training"""
    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Tạo datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    return train_dataset, test_dataset

def tokenize_function(examples, tokenizer):
    """Tokenize text"""
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

def compute_metrics(eval_pred):
    """Tính metrics cho evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def add_manual_training_data(conn):
    """Thêm dữ liệu training thủ công cho các trường hợp cơ bản"""
    cur = conn.cursor()
    
    # Dữ liệu training mẫu cho tiếng Việt
    training_samples = [
        # Tiêu cực
        ("tôi muốn chết", "TIÊU CỰC"),
        ("Tôi bị ngu", "TIÊU CỰC"),
        ("Món ăn này dở quá", "TIÊU CỰC"),
        ("Mệt mỏi quá", "TIÊU CỰC"),
        ("Tôi buồn vì thất bại", "TIÊU CỰC"),
        ("Thật tệ hại", "TIÊU CỰC"),
        ("Không thích cái này", "TIÊU CỰC"),
        ("Quá tồi tệ", "TIÊU CỰC"),
        ("Thất vọng quá", "TIÊU CỰC"),
        ("Chán ghê", "TIÊU CỰC"),
        
        # Tích cực
        ("Tôi rất vui", "TÍCH CỰC"),
        ("Món này ngon tuyệt", "TÍCH CỰC"),
        ("Tuyệt vời quá", "TÍCH CỰC"),
        ("Tôi yêu điều này", "TÍCH CỰC"),
        ("Quá đỉnh", "TÍCH CỰC"),
        ("Xuất sắc", "TÍCH CỰC"),
        ("Tôi hạnh phúc", "TÍCH CỰC"),
        ("Thật tuyệt", "TÍCH CỰC"),
        ("Tốt lắm", "TÍCH CỰC"),
        ("Hoàn hảo", "TÍCH CỰC"),
        
        # Trung lập
        ("Tôi là Đạt", "TRUNG LẬP"),
        ("Hôm nay thứ hai", "TRUNG LẬP"),
        ("Cái này là gì", "TRUNG LẬP"),
        ("Được đấy", "TRUNG LẬP"),
        ("Bình thường", "TRUNG LẬP"),
    ]
    
    from datetime import datetime
    for text, sentiment in training_samples:
        ts = datetime.utcnow().isoformat(sep=' ')
        cur.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)', 
                   (text, sentiment, ts))
    
    conn.commit()
    print(f"Đã thêm {len(training_samples)} mẫu training vào database")

def train_model():
    """Main training function"""
    print("=" * 50)
    print("BẮT ĐẦU FINE-TUNE MODEL SENTIMENT ANALYSIS")
    print("=" * 50)
    
    # Kiểm tra và thêm dữ liệu training
    if not os.path.exists(DB_PATH):
        print("❌ Database không tồn tại. Vui lòng chạy app trước.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sentiments")
    count = cur.fetchone()[0]
    
    if count < 20:
        print(f"⚠️  Chỉ có {count} mẫu. Đang thêm dữ liệu training mẫu...")
        add_manual_training_data(conn)
    
    conn.close()
    
    # Load data
    print("\n📊 Loading dữ liệu...")
    df = load_training_data()
    
    if len(df) < 10:
        print("❌ Cần ít nhất 10 mẫu để train. Vui lòng thêm nhiều dữ liệu hơn.")
        return
    
    # Prepare datasets
    print("\n🔧 Chuẩn bị datasets...")
    train_dataset, test_dataset = prepare_dataset(df)
    
    # Load tokenizer và model
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,  # 3 labels: tiêu cực (0), trung lập (1), tích cực (2)
        ignore_mismatched_sizes=True
    )
    
    # Tokenize datasets
    print("\n✂️  Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=10,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n🚀 Bắt đầu training...")
    print("=" * 50)
    trainer.train()
    
    # Evaluate
    print("\n📈 Đánh giá model...")
    results = trainer.evaluate()
    print("\nKết quả:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    print(f"\n💾 Lưu model vào {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("✅ HOÀN THÀNH! Model đã được fine-tune và lưu thành công.")
    print(f"📁 Model được lưu tại: {OUTPUT_DIR}")
    print("\n💡 Khởi động lại app để sử dụng model mới!")
    print("=" * 50)

if __name__ == "__main__":
    train_model()
