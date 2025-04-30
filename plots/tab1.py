import pandas as pd
import numpy as np 

METRIC = "recall@1"
METHODS = ["sparse-vit-BoQ", "sparse-vit-MixVPR", "sparse-vit-CLS"]
BASELINES = ["DINOv2-SALAD", "CosPlace", "EigenPlaces", "DINOv2-BoQ"]

# ----------------------- LaTeX table for RSS paper -------------------------
from textwrap import dedent

# ---------------------------- Load data ------------------------------------
results_df   = pd.read_csv("results/results.csv")
baselines_df = pd.read_csv("results/baselines.csv")

# ───────────────────────── constants ──────────────────────────
APPEAR_SPLITS  = ["SVOX-snow", "SVOX-rain", "SVOX-overcast",
                  "SVOX-night", "SVOX-sun"]

SPARSE_HEADS   = ["BoQ", "MixVPR", "CLS", "SALAD"]   # heads to show
BASELINES      = ["DINOv2-SALAD", "CosPlace",
                  "EigenPlaces", "DINOv2-BoQ"]

TARGET_SPARS   = 0.4          # 40 % sparsity
ATOL           = 1e-4         # float tolerance

# desired row order in the final table
ROW_ORDER = (
    [f"sparse-ViT-{h}-{int(TARGET_SPARS*100)}%" for h in SPARSE_HEADS]
    + BASELINES
)

# ───────────────────── sparse-ViT rows (0.4) ──────────────────
sparse_rows = (
    results_df
      .loc[
          # only appearance splits
          lambda d: d["dataset"].isin(APPEAR_SPLITS)
          # sparsity == 0.4 (within tolerance)
          & np.isclose(d["sparsity"].astype(float), TARGET_SPARS, atol=ATOL)
          # model name starts with sparse-vit- + wanted head
          & d["model"].isin([f"sparse-vit-{h}" for h in SPARSE_HEADS]),
          ["model", "dataset", "recall@1"]
      ]
      .assign(
          Method=lambda d:
              d["model"].str.replace("sparse-vit-", "sparse-ViT-", regex=False)
                         .add(f"-{int(TARGET_SPARS*100)}%")
      )
      .drop(columns="model")
)

# ───────────────────── baseline rows ──────────────────────────
base_rows = (
    baselines_df
      .loc[
          lambda d: d["dataset"].isin(APPEAR_SPLITS)
          & d["model"].isin(BASELINES),
          ["model", "dataset", "recall@1"]
      ]
      .rename(columns={"model": "Method"})
)

# ───────────────────── build the table ────────────────────────
table_df = (
    pd.concat([sparse_rows, base_rows], axis=0)
      .pivot_table(index="Method", columns="dataset", values="recall@1")
      .reindex(index=ROW_ORDER)        # enforce row order
      .loc[:, APPEAR_SPLITS]           # enforce column order
      .round(1)                        # one decimal place
)

# ───────────────────── export LaTeX ───────────────────────────
latex = table_df.to_latex(
    column_format="l" + "c"*len(APPEAR_SPLITS),  # l|ccccc
    bold_rows=True,
    caption=(
        "Recall@1 on appearance-change SVOX splits. "
        "Sparse-ViT heads are pruned to 40\\% sparsity; baselines are unpruned."
    ),
    label="tab:svox_appearance_sparse40",
    multicolumn=False,
    multicolumn_format="c",
    escape=False            # keep en-dashes
)

print(r"""
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{6pt}
""" + latex + r"""
\end{table}
""")