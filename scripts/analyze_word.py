import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 1. Dynamic Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Cek lokasi file dataset (Mencoba mlbb_reviews_raw.csv atau feedback.csv)
PRIMARY_CSV = os.path.join(PROJECT_ROOT, "data", "processed_reviews.csv")
FALLBACK_CSV = os.path.join(PROJECT_ROOT, "data", "feedback.csv")

if os.path.exists(PRIMARY_CSV):
    DATA_PATH = PRIMARY_CSV
elif os.path.exists(FALLBACK_CSV):
    DATA_PATH = FALLBACK_CSV
else:
    raise FileNotFoundError("File dataset tidak ditemukan di folder data/!")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Stopwords Bahasa Indonesia & Kosa Kata Umum Game (Diabaikan agar tidak mengotori hasil)
STOPWORDS_ID = [
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan", "saya",
    "ada", "bisa", "tidak", "gak", "ga", "ngga", "nggak", "game", "ml", "mlbb", "mobile",
    "legends", "bang", "nya", "aja", "biar", "karena", "kalo", "kalau", "jadi", "udah",
    "sudah", "sama", "tapi", "akan", "juga", "atau", "hanya", "sangat", "mohon", "tolong",
    "please", "plis", "dong", "deh", "kan", "kok", "moonton", "dev", "developer"
]

def get_top_ngrams(corpus, ngram_range=(2, 2), top_n=15):
    """Ekstraksi N-Gram teratas dari sekumpulan teks"""
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=STOPWORDS_ID,
        min_df=2
    )
    try:
        X = vectorizer.fit_transform(corpus)
        words = vectorizer.get_feature_names_out()
        frequencies = X.toarray().sum(axis=0)
        
        df_ngram = pd.DataFrame({
            "phrases": words,
            "frequency": frequencies
        }).sort_values(by="frequency", ascending=False)
        
        return df_ngram.head(top_n)
    except ValueError:
        # Menangani jika corpus terlalu sedikit/kosong
        return pd.DataFrame(columns=["phrases", "frequency"])

def run_keyword_analysis():
    print("=" * 65)
    print(" 🔍 N-GRAM & KEYWORD EXTRACTION FOR MLBB FEEDBACK")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    print(f"\n[1] Dataset berhasil dimuat dari: {DATA_PATH}")
    print(f"    - Total Data: {len(df)} baris")

    # Identifikasi kolom teks & rating
    text_col = "normalized_text" if "normalized_text" in df.columns else (
        "clean_text" if "clean_text" in df.columns else "comment"
    )
    rating_col = "score" if "score" in df.columns else "rating"

    print(f"    - Kolom Teks Digunakan   : '{text_col}'")
    print(f"    - Kolom Rating Digunakan : '{rating_col}'")

    # Bersihkan nilai NaN
    corpus_all = df[text_col].dropna().astype(str)
    corpus_neg = df[df[rating_col] <= 2][text_col].dropna().astype(str)
    corpus_pos = df[df[rating_col] >= 4][text_col].dropna().astype(str)

    # 3. Analisis Bigram (2 Kata) Keluhan Utama (Rating 1-2)
    print("\n[2] 🔴 TOP 50 FRASA KELUHAN (RATING 1-2 - BIGRAM):")
    top_neg_bigrams = get_top_ngrams(corpus_neg, ngram_range=(2, 2), top_n=50)
    for i, row in enumerate(top_neg_bigrams.itertuples(), 1):
        print(f"    {i:2d}. {row.phrases:<25} ({row.frequency}x muncul)")

    # 4. Analisis Trigram (3 Kata) Keluhan Utama (Rating 1-2)
    print("\n[3] 🔴 TOP 50 FRASA KELUHAN DETAIL (RATING 1-2 - TRIGRAM):")
    top_neg_trigrams = get_top_ngrams(corpus_neg, ngram_range=(3, 3), top_n=50)
    for i, row in enumerate(top_neg_trigrams.itertuples(), 1):
        print(f"    {i:2d}. {row.phrases:<30} ({row.frequency}x muncul)")

    # 5. Analisis Bigram Kepuasan (Rating 4-5)
    print("\n[4] 🟢 TOP 50 FRASA PUAS/APRESIASI (RATING 4-5 - BIGRAM):")
    top_pos_bigrams = get_top_ngrams(corpus_pos, ngram_range=(2, 2), top_n=50)
    for i, row in enumerate(top_pos_bigrams.itertuples(), 1):
        print(f"    {i:2d}. {row.phrases:<25} ({row.frequency}x muncul)")
    
    # 6. Analisis Trigram Kepuasan (Rating 4-5)
    print("\n[4] 🟢 TOP 50 FRASA PUAS/APRESIASI (RATING 4-5 - TRIGRAM):")
    top_pos_trigrams = get_top_ngrams(corpus_pos, ngram_range=(3, 3), top_n=50)
    for i, row in enumerate(top_pos_trigrams.itertuples(), 1):
        print(f"    {i:2d}. {row.phrases:<25} ({row.frequency}x muncul)")

    # 6. Simpan Laporan ke File CSV
    report_path = os.path.join(OUTPUT_DIR, "ngram_summary.csv")
    top_neg_bigrams["category_type"] = "Negative Feedback (Bigram)"
    top_neg_trigrams["category_type"] = "Negative Feedback (Trigram)"
    top_pos_bigrams["category_type"] = "Positive Feedback (Bigram)"
    top_pos_trigrams["category_type"] = "Positive Feedback (Trigram)"

    combined_report = pd.concat([top_neg_bigrams, top_neg_trigrams, top_pos_bigrams, top_pos_trigrams])
    combined_report.to_csv(report_path, index=False)

    print(f"\n✅ Laporan lengkap berhasil disimpan ke: {report_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_keyword_analysis()