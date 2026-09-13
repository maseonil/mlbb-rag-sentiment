import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config style visualisasi
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Set path dinamis relatif terhadap lokasi script ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "feedback.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# Buat folder output visualisasi jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda():
    print("=" * 60)
    print(" 📊 EXPLORATORY DATA ANALYSIS (EDA) - CUSTOMER FEEDBACK")
    print("=" * 60)

    # 1. Load Dataset
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH)
    print(f"\n[1] Dataset berhasil dimuat!")
    print(f"    - Total Baris : {len(df)}")
    print(f"    - Kolom       : {list(df.columns)}")

    # 2. Ringkasan Statistik Utama
    print("\n[2] Ringkasan Statistik:")
    avg_rating = df["rating"].mean()
    rating_counts = df["rating"].value_counts().sort_index()
    category_counts = df["category"].value_counts()
    
    print(f"    - Rata-rata Rating : {avg_rating:.2f} / 5.0")
    print("\n    - Distribusi Rating:")
    for rating, count in rating_counts.items():
        pct = (count / len(df)) * 100
        print(f"      ⭐ {rating}: {count} ulasan ({pct:.1f}%)")
        
    print("\n    - Distribusi Kategori Keluhan:")
    for cat, count in category_counts.items():
        cat_avg = df[df["category"] == cat]["rating"].mean()
        print(f"      • {cat:<15}: {count:<3} ulasan | Avg Rating: {cat_avg:.2f}")

    # 3. Analisis Panjang Teks Komentar
    df["comment_length"] = df["comment"].apply(lambda x: len(str(x).split()))
    print(f"\n    - Rata-rata Panjang Komentar: {df['comment_length'].mean():.1f} kata")

    # 4. Generate & Save Visualisasi Grafik
    print("\n[3] Memproses & Menyimpan Grafik Visualisasi...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Distribusi Rating
    sns.countplot(
        data=df, 
        x="rating", 
        ax=axes[0], 
        palette="viridis", 
        hue="rating", 
        legend=False
    )
    axes[0].set_title("Distribusi Rating Pelanggan", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rating (1-5)")
    axes[0].set_ylabel("Jumlah Ulasan")

    # Plot 2: Rata-rata Rating per Kategori
    avg_cat_df = df.groupby("category")["rating"].mean().reset_index().sort_values(by="rating")
    sns.barplot(
        data=avg_cat_df, 
        x="rating", 
        y="category", 
        ax=axes[1], 
        palette="mako", 
        hue="category", 
        legend=False
    )
    axes[1].set_title("Rata-rata Rating berdasarkan Kategori", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Rata-rata Rating")
    axes[1].set_ylabel("Kategori")
    axes[1].set_xlim(1, 5)

    plt.tight_layout()
    
    # Save Plot
    output_plot_path = os.path.join(OUTPUT_DIR, "eda_summary.png")
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"    ✅ Grafik berhasil disimpan ke: {output_plot_path}")
    print("\n" + "=" * 60)
    print(" 🎉 PROSES EDA SELESAI!")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()