🚀 Customer Feedback Analytics & RAG Engine

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white)
![Gemini API](https://img.shields.io/badge/AI-Gemini_3.6_Flash-8E44AD?style=flat&logo=google&logoColor=white)

Aplikasi analitik end-to-end untuk pemrosesan feedback pelanggan berbasis **FastAPI**, **Scikit-Learn** (Sentimen Analisis), **ChromaDB** (Vector Store), dan **Google Gemini API** (Retrieval-Augmented Generation).

---

## 🏗️ Arsitektur Sistem

```text
[User / Client]
       │
       ▼
[FastAPI Engine (Port 8000)]
   ├── /api/v1/analytics/predict-sentiment ──> [Logistic Regression + TF-IDF]
   ├── /api/v1/analytics/summary           ──> [Pandas Data Summarizer]
   └── /api/v1/rag/ask                     ──> [ChromaDB Vector Store]
                                                     │ (Semantic Query)
                                                     ▼
                                              [Gemini 3.6-Flash LLM]