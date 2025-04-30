import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# --------------------------- Configuration ---------------------------------
DATASET   = "Pitts30k"
METRIC    = "recall@1"
MEM_COL   = "memory"        # assumed to be in MB; convert below
SPARSE_RE = re.compile(r"sparse-vit-(?P<head>[A-Za-z]+)(?:-(?P<sparsity>\d+))?")

METHOD_HEADS = ["BoQ", "MixVPR", "CLS"]
HEAD_MARKERS = {"BoQ": "o", "MixVPR": "s", "CLS": "^"}      # shape by retrieval head
BASELINES    = ["DINOv2-SALAD", "CosPlace", "EigenPlaces", "DINOv2-BoQ"]
BASE_MARKERS = ["D", "P", "X", "v"]

# ---------------------------- Load data ------------------------------------
results_df   = pd.read_csv("results/results.csv")
baselines_df = pd.read_csv("results/baselines.csv")

# Canonical units: convert MB → GB for readability
for df in (results_df, baselines_df):
    if MEM_COL in df.columns:
        df[MEM_COL] = df[MEM_COL] / 1024.0          # MB → GB

# ---------------------- Extract sparsity information -----------------------
def get_sparse_meta(row):
    m = SPARSE_RE.fullmatch(row["model"])
    if m:
        sparsity = (100 - int(m["sparsity"])) if m["sparsity"] else 0
        return pd.Series({"head": m["head"], "sparsity": sparsity})
    return pd.Series({"head": None, "sparsity": None})

sparse_meta = results_df.apply(get_sparse_meta, axis=1)
results_df  = pd.concat([results_df, sparse_meta], axis=1)

# Keep only the requested dataset
results_df   = results_df.query("dataset == @DATASET")
baselines_df = baselines_df.query("dataset == @DATASET")

# --------------------------- Plotting --------------------------------------
plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# --- sparse-ViT points ------------------------------------------------------
cmap   = plt.get_cmap("viridis")
norm   = mpl.colors.Normalize(vmin=0, vmax=90)     # 0% (dense)  →  90% sparse

for head in METHOD_HEADS:
    df_head = results_df.query("head == @head")
    sc = ax.scatter(
        df_head[MEM_COL],
        df_head[METRIC],
        c=df_head["sparsity"],
        cmap=cmap,
        norm=norm,
        marker=HEAD_MARKERS[head],
        s=60,
        edgecolors="black",
        linewidths=0.5,
        label=f"sparse-ViT-{head}"
    )

# colourbar keyed to sparsity level
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("Sparsity removed (%)", rotation=270, labelpad=15)

# --- Baseline curves -------------------------------------------------------
for b, mk in zip(BASELINES, BASE_MARKERS):
    df_b = baselines_df.query("model == @b")
    ax.scatter(
        df_b[MEM_COL],
        df_b[METRIC],
        marker=mk,
        s=70,
        facecolors="none",
        edgecolors="grey",
        linewidths=1.2,
        label=f"{b} (baseline)"
    )

# -------------------------- Axis cosmetics ---------------------------------
ax.set_xlabel("Memory footprint (GB)")
ax.set_ylabel("Recall@1")
ax.set_title(f"{DATASET}: Memory vs. Recall@1")
ax.set_xscale("log")          # optional: comment out if memory range is narrow
ax.grid(True, linestyle=":", linewidth=0.7)
ax.legend(fontsize=8, frameon=True, edgecolor="0.8")
fig.tight_layout()

# --------------------------- Save & show -----------------------------------
Path("plots/images").mkdir(parents=True, exist_ok=True)
fig.savefig("plots/images/memory_vs_recall.png", bbox_inches="tight", dpi=300)
plt.close(fig)
