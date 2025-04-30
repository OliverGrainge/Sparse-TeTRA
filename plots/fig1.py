import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---------------------------------------------------------
DATASET = "Pitts30k"
METRIC = "recall@1"
METHODS = ["sparse-vit-BoQ", "sparse-vit-MixVPR", "sparse-vit-CLS"]
BASELINES = ["DINOv2-SALAD", "CosPlace", "EigenPlaces", "DINOv2-BoQ"]

try:
    # --- Load Data -------------------------------------------------------------
    results_df = pd.read_csv("results/results.csv")
    baselines_df = pd.read_csv("results/baselines.csv")

    # Matplotlib style for a clean, white background with gridlines
    plt.style.use("seaborn-whitegrid")

    # --- Figure Setup ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)  # 6×4 inches at 300 dpi

    # Define a consistent color and marker cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v"]

    # --- Plot Methods ----------------------------------------------------------
    for i, method in enumerate(METHODS):
        try:
            df = results_df.query("model == @method and dataset == @DATASET")
            if len(df) == 0:
                print(f"No data found for method {method} on dataset {DATASET}")
                continue
                
            ax.plot(
                df["flops"],
                df[METRIC],
                label=method,
                marker=markers[i % len(markers)],
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                color=colors[i]
            )
        except Exception as e:
            print(f"Error plotting method {method}: {str(e)}")

    # --- Plot Baselines --------------------------------------------------------
    for j, base in enumerate(BASELINES, start=len(METHODS)):
        try:
            df = baselines_df.query("model == @base and dataset == @DATASET")
            if len(df) == 0:
                print(f"No data found for baseline {base} on dataset {DATASET}")
                continue
                
            ax.plot(
                df["flops"],
                df[METRIC],
                label=base + " (baseline)",
                marker=markers[j % len(markers)],
                linestyle="--",
                linewidth=1.2,
                markersize=6,
                color=colors[j]
            )
        except Exception as e:
            print(f"Error plotting baseline {base}: {str(e)}")

    # --- Axes & Labels ---------------------------------------------------------
    ax.set_xscale("log")                        # FLOPs typically span orders of magnitude
    ax.set_xlabel("FLOPs (×10⁹)", fontsize=12)
    ax.set_ylabel("Recall@1",      fontsize=12)
    ax.set_title(f"Performance on {DATASET}", fontsize=14, pad=10)

    # Ticks & Grid -------------------------------------------------------------
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8)

    # Legend -------------------------------------------------------------------
    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        edgecolor="0.8"
    )

    # Tight layout & save ------------------------------------------------------
    fig.tight_layout(pad=1.0)
    fig.savefig("plots/images/fig1.png", bbox_inches="tight")
    plt.close(fig)

except FileNotFoundError as e:
    print(f"Error: Could not find data files - {str(e)}")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
