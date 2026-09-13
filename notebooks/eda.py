import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set style visualisasi
sns.set_theme(style="whitegrid")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "feedback.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset tidak ditemukan di {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Memuat {len(df)} baris data untuk analisis EDA.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Grafik Distribusi Rating
    if "rating" in df.columns:
        sns.countplot(data=df, x="rating", palette="viridis", ax=axes[0])
        axes[0].set_title("Distribusi Rating Pelanggan", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Rating (1-5)")
        axes[0].set_ylabel("Jumlah Ulasan")

    # 2. Bar Chart Kategori Keluhan
    if "category" in df.columns:
        sns.countplot(data=df, y="category", palette="magma", order=df["category"].value_counts().index, ax=axes[1])
        axes[1].set_title("Distribusi Kategori Feedback", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Jumlah")
        axes[1].set_ylabel("Kategori")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_summary.png"), dpi=300)
    plt.close()

    # 3. WordCloud Kata Terbanyak
    comment_col = "comment" if "comment" in df.columns else df.columns[0]
    text_data = " ".join(df[comment_col].dropna().astype(str))

    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Dark2").generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Visualisasi Kata Terbanyak (WordCloud)", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"), dpi=300)
    plt.close()

    print(f"✅ Visualisasi berhasil disimpan di folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_eda()