import os
import sys
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Disable Telemetry & Suppress Warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NO_ANALYTICS"] = "True"

logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

load_dotenv()

# Setup Path Impor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.rag_engine import get_rag_chain
except ImportError:
    from rag_engine import get_rag_chain

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.pkl")
VEC_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "feedback.csv")

# Load ML Models
sentiment_model = None
tfidf_vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    try:
        sentiment_model = joblib.load(MODEL_PATH)
        tfidf_vectorizer = joblib.load(VEC_PATH)
    except Exception as e:
        logging.error(f"Gagal memuat model ML: {e}")

# Inisialisasi RAG Chain
rag_chain = get_rag_chain()

app = FastAPI(
    title="Customer Feedback Analytics & RAG Engine",
    version="1.0.0",
    description="API untuk analisis sentimen, statistik feedback, dan QA RAG berbasis Gemini 3.6-flash."
)

class QuestionRequest(BaseModel):
    question: str

class SentimentRequest(BaseModel):
    comment: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Customer Feedback Analytics & RAG Engine",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/analytics/summary")
def get_analytics_summary():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="Dataset feedback.csv tidak ditemukan.")
    
    df = pd.read_csv(CSV_PATH)
    avg_rating = float(df["rating"].mean()) if "rating" in df.columns else 0.0
    total_feedback = len(df)
    category_dist = df["category"].value_counts().to_dict() if "category" in df.columns else {}

    return {
        "total_feedback": total_feedback,
        "average_rating": round(avg_rating, 2),
        "category_distribution": category_dist
    }

@app.post("/api/v1/analytics/predict-sentiment")
def predict_sentiment(payload: SentimentRequest):
    if not sentiment_model or not tfidf_vectorizer:
        raise HTTPException(
            status_code=500, 
            detail="Model ML belum dilatih. Jalankan `python src/train_sentiment.py` terlebih dahulu."
        )
    
    if not payload.comment.strip():
        raise HTTPException(status_code=400, detail="Komentar tidak boleh kosong.")

    vec_text = tfidf_vectorizer.transform([payload.comment])
    prediction = int(sentiment_model.predict(vec_text)[0])
    probabilities = sentiment_model.predict_proba(vec_text)[0]
    confidence = float(max(probabilities))

    sentiment_label = "POSITIF" if prediction == 1 else "NEGATIF"

    return {
        "comment": payload.comment,
        "sentiment": sentiment_label,
        "confidence_score": round(confidence, 4),
        "is_positive": prediction == 1
    }

@app.post("/api/v1/rag/ask")
def ask_rag(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
    
    response = rag_chain.invoke({"input": payload.question})
    return {
        "question": payload.question,
        "answer": response["answer"]
    }