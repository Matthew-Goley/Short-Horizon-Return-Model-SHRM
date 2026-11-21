import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results.csv"
OUT_DIR = ROOT / "results" / "figures"

# Column order of the write call in scripts/train.py (results.csv has no header).
COLUMNS = [
    "epoch", "best_val_nll", "train_nll", "val_nll", "val_abs_mu", "val_sigma",
    "val_conf", "val_pred_pos_rate", "val_snr", "edge_score", "dir_acc",
]


def load_runs(path):
    rows = []
    with open(path, newline="") as f:
        for rec in csv.reader(f):
            if rec:
                row = dict(zip(COLUMNS, map(float, rec)))
                row["epoch"] = int(row["epoch"])
                rows.append(row)

    runs = []
    for row in rows:
        if not runs or row["epoch"] <= runs[-1][-1]["epoch"]:
            runs.append([])
        runs[-1].append(row)
    return rows, runs


def plot_metric(runs, key, ylabel, filename, ref=None, ref_label=None):
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    ax.set_facecolor("white")

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, run in enumerate(runs):
        ax.plot(
            [r["epoch"] for r in run],
            [r[key] for r in run],
            marker=markers[i % len(markers)],
            markersize=8,
            linewidth=1.5,
            label=f"Run {i + 1} ({len(run)} row{'s' if len(run) != 1 else ''})",
        )

    if ref is not None:
        ax.axhline(ref, color="black", linestyle="--", linewidth=1.2, label=ref_label)

    ax.set_xlabel("Epoch (count)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    rows, runs = load_runs(CSV_PATH)

    print("Columns:", ", ".join(COLUMNS))
    for i, run in enumerate(runs, 1):
        for row in run:
            print(f"run {i}: " + ", ".join(f"{k}={row[k]}" for k in COLUMNS))
    print(f"{len(rows)} rows, {len(runs)} runs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric(
        runs, "val_nll",
        "Validation Gaussian NLL (nats)",
        "val_nll_per_epoch.png",
    )
    plot_metric(
        runs, "dir_acc",
        "Validation directional accuracy (fraction)",
        "dir_acc_per_epoch.png",
        ref=0.5, ref_label="0.5 = coin flip",
    )
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
