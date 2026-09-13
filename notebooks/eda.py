import os
import pandas as pd
import matplotlib
# Menggunakan backend Agg agar aman di semua lingkungan (non-GUI)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set_theme(style="whitegrid")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "feedback.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda():
    if not os.path.exists(DATA_PATH):
        print(f"❌ File tidak ditemukan di: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    # Normalisasi nama kolom ke huruf kecil & hapus spasi liar
    df.columns = df.columns.str.lower().str.strip()
    print(f"✅ Data dimuat ({len(df)} baris). Kolom terdeteksi: {list(df.columns)}")

    # 1. Plot Ringkasan EDA (Rating & Kategori)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "rating" in df.columns:
        sns.countplot(data=df, x="rating", palette="viridis", ax=axes[0])
        axes[0].set_title("Distribusi Rating Pelanggan", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Rating")
        axes[0].set_ylabel("Jumlah")
    else:
        axes[0].text(0.5, 0.5, "Kolom 'rating' tidak ditemukan", ha='center')

    if "category" in df.columns:
        sns.countplot(data=df, y="category", palette="magma", order=df["category"].value_counts().index, ax=axes[1])
        axes[1].set_title("Distribusi Kategori Feedback", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Jumlah")
        axes[1].set_ylabel("Kategori")
    else:
        axes[1].text(0.5, 0.5, "Kolom 'category' tidak ditemukan", ha='center')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "eda_summary.png")
    
    # PERHATIAN: savefig HARUS sebelum close/show
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved EDA Plot: {plot_path}")

    # 2. Plot WordCloud
    comment_col = "comment" if "comment" in df.columns else df.columns[0]
    text_data = " ".join(df[comment_col].dropna().astype(str))

    if text_data.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Dark2").generate(text_data)
        fig_wc = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Visualisasi Kata Terbanyak (WordCloud)", fontsize=14, fontweight="bold")
        
        wc_path = os.path.join(OUTPUT_DIR, "wordcloud.png")
        plt.savefig(wc_path, dpi=300, bbox_inches='tight')
        plt.close(fig_wc)
        print(f"✅ Saved WordCloud: {wc_path}")

if __name__ == "__main__":
    run_eda()