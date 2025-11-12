import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_data, clean_and_engineer

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = load_data()
df = clean_and_engineer(df)

# 1️⃣ Price distribution
numeric_cols = df.select_dtypes("number").columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols)//3 + 1, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(col, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "all_numeric_histograms.png"), dpi=300)
plt.close()

# 2️⃣ Boxplots
numeric_cols = [col for col in df.select_dtypes("number").columns if df[col].nunique() > 2]
df[numeric_cols].plot(kind="box", subplots=True, layout=(len(numeric_cols)//3+1,3),
                      figsize=(20,15), color="lightblue", patch_artist=True)
plt.suptitle("Boxplots of Numeric Columns", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "boxplots.png"))
plt.close()

# 3️⃣ Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
plt.close()

print(f"✅ All EDA plots saved to {FIG_DIR}")
