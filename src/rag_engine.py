import os
import pandas as pd
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
VECTOR_DIR = os.path.join(PROJECT_ROOT, "chroma_db_local")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "feedback.csv")

class RAGEngine:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name="mlbb_feedback")
        
        if self.collection.count() == 0 and os.path.exists(CSV_PATH):
            self._populate_vector_store()
            
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _populate_vector_store(self):
        df = pd.read_csv(CSV_PATH)
        comment_col = "comment" if "comment" in df.columns else df.columns[0]
        
        documents = df[comment_col].dropna().astype(str).tolist()
        if documents:
            ids = [f"doc_{idx}" for idx in range(len(documents))]
            metadatas = [{"category": str(df.iloc[idx]["category"])} if "category" in df.columns else {"category": "General"} for idx in range(len(documents))]
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def invoke(self, input_data: dict) -> dict:
        question = input_data.get("input", "").strip()
        if not question:
            return {"answer": "Pertanyaan tidak boleh kosong."}
        
        if not self.client:
            return {"answer": "GEMINI_API_KEY belum dikonfigurasi di file .env."}

        results = self.collection.query(query_texts=[question], n_results=3)
        retrieved_docs = results["documents"][0] if results["documents"] and len(results["documents"]) > 0 else []
        
        context_str = "\n".join([f"- {doc}" for doc in retrieved_docs]) if retrieved_docs else "Tidak ada ulasan relevan."

        prompt = (
            "Anda adalah AI Analyst yang bertugas menganalisis keluhan dan feedback pelanggan Mobile Legends.\n"
            "Jawab pertanyaan pengguna secara ringkas, jelas, dan faktual berdasarkan konteks ulasan berikut.\n\n"
            f"Konteks Ulasan Pelanggan:\n{context_str}\n\n"
            f"Pertanyaan: {question}\n"
            "Jawaban:"
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            return {"answer": response.text}
        except Exception as e:
            return {"answer": f"Gagal memproses RAG Engine: {str(e)}"}

def get_rag_chain():
    return RAGEngine()