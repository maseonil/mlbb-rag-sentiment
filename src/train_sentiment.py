import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Dynamic Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "feedback.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_sentiment_model():
    print("=" * 60)
    print(" 🤖 TRAINING LOGISTIC REGRESSION - SENTIMENT ANALYSIS")
    print("=" * 60)

    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    print(f"\n[1] Data berhasil dimuat ({len(df)} baris).")

    # 2. Binarization Target (Rating >= 4: Positif/1, Rating < 4: Negatif/0)
    df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    
    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    print(f"    - Sentiment Positif (1) : {pos_count} data")
    print(f"    - Sentiment Negatif (0) : {neg_count} data")

    X = df["comment"]
    y = df["label"]

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Extraction via TF-IDF
    print("\n[2] Ekstraksi Fitur Teks dengan TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Training Logistic Regression
    print("\n[3] Melatih Model Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_vec, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("\n[4] Hasil Evaluasi Model:")
    print(f"    - Akurasi Model: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Negatif (0)", "Positif (1)"]))

    # 7. Save Model & Vectorizer Artifacts
    model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"✅ Model berhasil disimpan di      : {model_path}")
    print(f"✅ Vectorizer berhasil disimpan di : {vec_path}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    train_sentiment_model()