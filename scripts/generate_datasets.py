import os
import pandas as pd

# 1. Path Setup Dinamis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Lokasi pencarian file dataset input (mencari processed_reviews.csv atau mlbb_reviews_raw.csv)
POSSIBLE_INPUT_PATHS = [
    os.path.join(PROJECT_ROOT, "processed_reviews.csv"),
    os.path.join(PROJECT_ROOT, "data", "processed_reviews.csv"),
    os.path.join(PROJECT_ROOT, "data", "mlbb_reviews_raw.csv"),
    os.path.join(PROJECT_ROOT, "mlbb_reviews_raw.csv")
]

OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "feedback.csv")

def find_input_file():
    """Mencari file CSV input di berbagai lokasi umum"""
    for path in POSSIBLE_INPUT_PATHS:
        if os.path.exists(path):
            return path
    return None

def categorize_mlbb_review(row):
    """
    Fungsi klasifikasi berbasis skor kata kunci & slang khas MLBB.
    Memeriksa teks ulasan (content & normalized_text) secara presisi.
    """
    text = f"{row.get('content', '')} {row.get('normalized_text', '')}".lower()

    # Bobot kata kunci berdasarkan kategori
    scores = {
        "Matchmaking": 0,
        "Server & Network": 0,
        "App Stability & Bug": 0,
        "Gameplay & Hero Balance": 0,
        "Monetization & Event": 0
    }

    # 1. Matchmaking
    for kw in ['dark system', 'dark sistem', 'lose streak', 'losestrek', 'tim bot', 'tim beban', 'troll', 'afk', 'bocil', 'bot', 'beban', 'solo rank', 'win streak']:
        if kw in text:
            scores["Matchmaking"] += (3 if kw in ['dark system', 'dark sistem', 'lose streak', 'losestrek', 'tim bot', 'tim beban', 'troll', 'afk'] else 1)

    # 2. Server & Network
    for kw in ['ping', 'lag', 'jaringan', 'sinyal', 'reconnect', 'menghubungkan', 'lambat', 'patah', 'ms', 'delay', 'disconnect', 'ping merah']:
        if kw in text:
            scores["Server & Network"] += (3 if kw in ['ping', 'lag', 'reconnect', 'menghubungkan', 'ms', 'patah', 'delay', 'ping merah'] else 1)

    # 3. App Stability & Bug
    for kw in ['bug', 'ngebug', 'crash', 'force close', 'layar hitam', 'blackscreen', 'stuck', 'loading', 'download', 'update', 'berat', 'panas', 'ram', 'keluar sendiri', 'hapus', 'memori']:
        if kw in text:
            scores["App Stability & Bug"] += (3 if kw in ['bug', 'ngebug', 'crash', 'force close', 'blackscreen', 'stuck', 'keluar sendiri'] else 1)

    # 4. Gameplay & Hero Balance
    for kw in ['hero', 'buff', 'nerf', 'skill', 'damage', 'meta', 'revamp', 'lifesteal', 'op', 'combo', 'gepeng', 'lemah', 'overpower']:
        if kw in text:
            scores["Gameplay & Hero Balance"] += (3 if kw in ['buff', 'nerf', 'revamp', 'lifesteal', 'op', 'overpower', 'skill combo'] else 1)

    # 5. Monetization & Event
    for kw in ['skin', 'gacha', 'diamond', 'dm', 'starlight', 'event', 'pelit', 'topup', 'top up', 'resale', 'undi', 'zonk', 'scam', 'mahal', 'gratis', 'ampas']:
        if kw in text:
            scores["Monetization & Event"] += (3 if kw in ['gacha', 'diamond', 'starlight', 'topup', 'top up', 'resale', 'zonk', 'scam', 'ampas'] else 1)

    max_score = max(scores.values())
    if max_score > 0:
        # Ambil kategori dengan skor tertinggi
        best_cat = [cat for cat, sc in scores.items() if sc == max_score][0]
        return best_cat
    
    return "General Feedback"

def adapt_dataset():
    print("=" * 65)
    print(" 🔄 ADAPTING MLBB DATASET TO STANDARDIZED FEEDBACK FORMAT")
    print("=" * 65)

    input_path = find_input_file()
    if not input_path:
        raise FileNotFoundError(
            "❌ File dataset MLBB tidak ditemukan!\n"
            "Pastikan file `processed_reviews.csv` atau `mlbb_reviews_raw.csv` "
            "berada di root folder proyek atau di folder `data/`."
        )

    print(f"\n[1] Membaca dataset dari: {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"    - Total Data Awal : {len(df_raw)} baris")

    # 2. Pemilihan Kolom Teks Utama (Prioritas: normalized_text -> clean_text -> content)
    if "normalized_text" in df_raw.columns:
        text_series = df_raw["normalized_text"].fillna(df_raw.get("content", ""))
    elif "clean_text" in df_raw.columns:
        text_series = df_raw["clean_text"].fillna(df_raw.get("content", ""))
    else:
        text_series = df_raw["content"]

    df_raw["final_comment"] = text_series

    # Hapus baris dengan komentar kosong
    df_raw = df_raw.dropna(subset=["final_comment"])
    df_raw = df_raw[df_raw["final_comment"].astype(str).str.strip() != ""]

    # 3. Proses Pengkategorian Teks
    print("\n[2] Mengategorikan ulasan berbasis kata kunci & slang MLBB...")
    categories = df_raw.apply(categorize_mlbb_review, axis=1)

    # 4. Pemetaan Kolom ke Format Baku Proyek
    print("\n[3] Memetakan kolom ke format `data/feedback.csv`...")
    df_adapted = pd.DataFrame({
        "review_id": df_raw["reviewId"] if "reviewId" in df_raw.columns else range(1, len(df_raw) + 1),
        "customer_name": df_raw["userName"] if "userName" in df_raw.columns else "Anonymous Player",
        "rating": df_raw["score"] if "score" in df_raw.columns else df_raw.get("rating", 3),
        "comment": df_raw["final_comment"],
        "category": categories
    })

    # 5. Simpan Hasil ke data/feedback.csv
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_adapted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n✅ Berhasil menyimpan {len(df_adapted)} baris data ke: {OUTPUT_CSV}")
    print("\n📊 Rincian Distribusi Kategori Baru:")
    print("-" * 45)
    for cat, count in df_adapted["category"].value_counts().items():
        pct = (count / len(df_adapted)) * 100
        print(f"  • {cat:<25}: {count:>4} ulasan ({pct:.1f}%)")
    print("-" * 45)
    print("=" * 65)

if __name__ == "__main__":
    adapt_dataset()