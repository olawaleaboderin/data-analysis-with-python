# ============================================================
# VISUALIZATION TEMPLATE 
# ============================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# 2. CONFIG (EDIT ONLY THIS)
# =========================
FILE_PATH = "clean_data.csv"

CATEGORY_COL = None   # e.g. "branch"
VALUE_COL = None      # e.g. "sales"
TIME_COL = None       # e.g. "month"

SAVE_DIR = "charts"
SAVE_PLOTS = True

# =========================
# 3. LOAD DATA
# =========================
df = pd.read_csv(FILE_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Auto-detect columns if not provided
if CATEGORY_COL is None:
    CATEGORY_COL = df.select_dtypes(include='object').columns[0]

if VALUE_COL is None:
    VALUE_COL = df.select_dtypes(include=np.number).columns[0]

# =========================
# 4. PREPARE ENVIRONMENT
# =========================
sns.set_style("whitegrid")

if SAVE_PLOTS:
    os.makedirs(SAVE_DIR, exist_ok=True)

# Generate dynamic colors
categories = df[CATEGORY_COL].dropna().unique()
palette = sns.color_palette("tab10", len(categories))
color_map = dict(zip(categories, palette))

# =========================
# 5. BAR CHART (AGGREGATE VIEW)
# =========================
print("📊 Generating bar chart...")

agg = df.groupby(CATEGORY_COL)[VALUE_COL].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
colors = [color_map.get(cat, "#333333") for cat in agg.index]

plt.bar(agg.index, agg.values, color=colors)
plt.title(f"Total {VALUE_COL} by {CATEGORY_COL}")
plt.xlabel(CATEGORY_COL)
plt.ylabel(VALUE_COL)
plt.xticks(rotation=45)

plt.tight_layout()

if SAVE_PLOTS:
    plt.savefig(f"{SAVE_DIR}/bar_chart.png", dpi=150)

plt.show()

# =========================
# 6. BOX PLOT (DISTRIBUTION)
# =========================
print("📊 Generating box plot...")

plt.figure(figsize=(12, 6))

sns.boxplot(
    x=CATEGORY_COL,
    y=VALUE_COL,
    data=df,
    palette=color_map
)

plt.title(f"Distribution of {VALUE_COL} by {CATEGORY_COL}")
plt.xticks(rotation=45)

plt.tight_layout()

if SAVE_PLOTS:
    plt.savefig(f"{SAVE_DIR}/box_plot.png", dpi=150)

plt.show()

# =========================
# 7. KDE PLOT (DENSITY SHAPE)
# =========================
print("📊 Generating KDE plot...")

plt.figure(figsize=(12, 6))

for cat in categories:
    subset = df[df[CATEGORY_COL] == cat][VALUE_COL]

    if subset.dropna().shape[0] > 1:
        sns.kdeplot(
            subset,
            label=str(cat),
            fill=True,
            alpha=0.2
        )

plt.title(f"Density Distribution of {VALUE_COL}")
plt.legend(title=CATEGORY_COL)

plt.tight_layout()

if SAVE_PLOTS:
    plt.savefig(f"{SAVE_DIR}/kde_plot.png", dpi=150)

plt.show()

# =========================
# 8. LINE CHART (TIME TREND)
# =========================
if TIME_COL and TIME_COL in df.columns:

    print("📊 Generating line chart...")

    # Try to convert to datetime safely
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='ignore')

    plt.figure(figsize=(12, 6))

    for cat in categories:
        subset = df[df[CATEGORY_COL] == cat].sort_values(TIME_COL)

        plt.plot(
            subset[TIME_COL],
            subset[VALUE_COL],
            label=str(cat),
            marker='o'
        )

    plt.title(f"{VALUE_COL} Trend over {TIME_COL}")
    plt.xlabel(TIME_COL)
    plt.ylabel(VALUE_COL)
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(f"{SAVE_DIR}/line_chart.png", dpi=150)

    plt.show()

# =========================
# 9. CORRELATION HEATMAP
# =========================
print("📊 Generating correlation heatmap...")

numeric_cols = df.select_dtypes(include=np.number)

if numeric_cols.shape[1] > 1:

    corr = numeric_cols.corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(f"{SAVE_DIR}/correlation_heatmap.png", dpi=150)

    plt.show()

# =========================
# 10. COMPLETION MESSAGE
# =========================
print("\n✅ All visualizations generated successfully")

if SAVE_PLOTS:
    print(f"📁 Charts saved in: {SAVE_DIR}/")