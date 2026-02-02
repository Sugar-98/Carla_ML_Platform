"""
Plot training log CSV.

Requirements (from user):
- Plot Train and Val together in the same figure for each metric.
- For class-wise metrics, plot all classes together in a single figure (per metric type).
- Adjust legend to be clean and readable.
- Additionally: plot train-val diff in separate figures.

Usage:
  python plot_log_csv.py --csv log.csv --outdir plots --show

Outputs:
  plots/loss.png
  plots/loss_diff.png
  plots/time_train_val.png
  plots/mIoU.png
  plots/mIoU_diff.png
  ...
  plots/precision_all_classes.png
  plots/precision_diff_all_classes.png
"""

import argparse
import os
import re
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CLASS_METRIC_RE = re.compile(r"^(precision|recall|f1|iou)_(\d+)_(train|val)$")
SPLIT_SUFFIX_RE = re.compile(r"^(.*)_(train|val)$")


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# Train/Val scalar plot
# ============================================================
def plot_train_val_scalar(
    df: pd.DataFrame,
    epoch_col: str,
    metric_base: str,
    outpath: str,
    title: str = None,
) -> None:
    x = df[epoch_col].to_numpy()

    col_train = f"{metric_base}_train"
    col_val = f"{metric_base}_val"

    if col_train not in df.columns or col_val not in df.columns:
        return

    y_train = pd.to_numeric(df[col_train], errors="coerce").to_numpy()
    y_val = pd.to_numeric(df[col_val], errors="coerce").to_numpy()

    # Filter out NaN values
    valid_indices = ~np.isnan(y_train) & ~np.isnan(y_val)
    x = x[valid_indices]
    y_train = y_train[valid_indices]
    y_val = y_val[valid_indices]

    fig = plt.figure(figsize=(9.5, 5.5))
    ax = fig.add_subplot(111)

    if len(x) > 0:  # Only plot if there is valid data
        ax.plot(x, y_train, label="Train")
        ax.plot(x, y_val, label="Val")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_base)
    ax.set_title(title or metric_base)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ============================================================
# Train/Val diff plot (scalar)
# ============================================================
def plot_diff_scalar(
    df: pd.DataFrame,
    epoch_col: str,
    metric_base: str,
    outpath: str,
):
    x = df[epoch_col].to_numpy()
    col_train = f"{metric_base}_train"
    col_val = f"{metric_base}_val"

    if col_train not in df.columns or col_val not in df.columns:
        return

    y_train = pd.to_numeric(df[col_train], errors="coerce").to_numpy()
    y_val = pd.to_numeric(df[col_val], errors="coerce").to_numpy()

    # Filter out NaN values
    valid_indices = ~np.isnan(y_train) & ~np.isnan(y_val)
    x = x[valid_indices]
    y_train = y_train[valid_indices]
    y_val = y_val[valid_indices]
    diff = y_train - y_val

    fig = plt.figure(figsize=(9.5, 5.5))
    ax = fig.add_subplot(111)

    if len(x) > 0:  # Only plot if there is valid data
        ax.plot(x, diff, label="Train - Val", color="purple")

    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{metric_base} diff")
    ax.set_title(f"{metric_base}: Train - Val")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ============================================================
# Special two-series plot (time)
# ============================================================
def plot_train_val_two_series(
    df: pd.DataFrame,
    epoch_col: str,
    train_col: str,
    val_col: str,
    outpath: str,
    ylabel: str,
    title: str,
) -> None:
    x = df[epoch_col].to_numpy()

    y_train = pd.to_numeric(df[train_col], errors="coerce").to_numpy()
    y_val = pd.to_numeric(df[val_col], errors="coerce").to_numpy()

    fig = plt.figure(figsize=(9.5, 5.5))
    ax = fig.add_subplot(111)

    ax.plot(x, y_train, label="Train")
    ax.plot(x, y_val, label="Val")

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ============================================================
# Class-wise train/val plot
# ============================================================
def plot_all_classes_train_val(
    df: pd.DataFrame,
    epoch_col: str,
    metric_name: str,
    class_ids: List[int],
    outpath: str,
) -> None:
    x = df[epoch_col].to_numpy()

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(class_ids))]

    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(111)

    for idx, c in enumerate(class_ids):
        col_train = f"{metric_name}_{c}_train"
        col_val = f"{metric_name}_{c}_val"
        if col_train not in df.columns or col_val not in df.columns:
            continue

        y_train = pd.to_numeric(df[col_train], errors="coerce").to_numpy()
        y_val = pd.to_numeric(df[col_val], errors="coerce").to_numpy()

        # Filter out NaN values
        valid_indices = ~np.isnan(y_train) & ~np.isnan(y_val)
        x_valid = x[valid_indices]
        y_train = y_train[valid_indices]
        y_val = y_val[valid_indices]

        if len(x_valid) > 0:  # Only plot if there is valid data
            ax.plot(x_valid, y_train, linestyle="-", color=colors[idx], label=f"C{c} Train")
            ax.plot(x_valid, y_val, linestyle="--", color=colors[idx], label=f"C{c} Val")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}: all classes (Train solid / Val dashed)")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
            ncol=2,
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Class-wise diff plot
# ============================================================
def plot_diff_classwise(
    df: pd.DataFrame,
    epoch_col: str,
    metric_name: str,
    class_ids: List[int],
    outpath: str,
):
    x = df[epoch_col].to_numpy()
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(class_ids))]

    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(111)

    for idx, c in enumerate(class_ids):
        col_train = f"{metric_name}_{c}_train"
        col_val = f"{metric_name}_{c}_val"
        if col_train not in df.columns or col_val not in df.columns:
            continue

        y_train = pd.to_numeric(df[col_train], errors="coerce").to_numpy()
        y_val = pd.to_numeric(df[col_val], errors="coerce").to_numpy()

        # Filter out NaN values
        valid_indices = ~np.isnan(y_train) & ~np.isnan(y_val)
        x_valid = x[valid_indices]
        y_train = y_train[valid_indices]
        y_val = y_val[valid_indices]
        diff = y_train - y_val

        if len(x_valid) > 0:  # Only plot if there is valid data
            ax.plot(x_valid, diff, color=colors[idx], label=f"C{c} diff")

    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{metric_name} diff")
    ax.set_title(f"{metric_name}: Train - Val (diff)")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
            ncol=2,
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV log file")
    parser.add_argument("--outdir", default="plots", help="Directory to save figures")
    parser.add_argument("--show", action="store_true", help="Show figures interactively (also saves)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        return 2

    safe_makedirs(args.outdir)

    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        print("[ERROR] CSV must contain 'epoch' column.", file=sys.stderr)
        return 2

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).reset_index(drop=True)
    epoch_col = "epoch"

    # --- Special paired metrics (time)
    if "time_train(h)" in df.columns and "time_val(h)" in df.columns:
        plot_train_val_two_series(
            df=df,
            epoch_col=epoch_col,
            train_col="time_train(h)",
            val_col="time_val(h)",
            outpath=os.path.join(args.outdir, "time_train_val.png"),
            ylabel="time (hours)",
            title="time (Train vs Val)",
        )

    # --- Scalar metrics
    bases: Dict[str, set] = {}
    for col in df.columns:
        m = SPLIT_SUFFIX_RE.match(col)
        if not m:
            continue
        base, split = m.group(1), m.group(2)
        bases.setdefault(base, set()).add(split)

    def is_classwise_base(base: str) -> bool:
        return bool(re.match(r"^(precision|recall|f1|iou)_\d+$", base))

    scalar_bases = sorted([b for b, splits in bases.items() if splits == {"train", "val"} and not is_classwise_base(b)])

    for base in scalar_bases:
        outpath = os.path.join(args.outdir, f"{base}.png")
        plot_train_val_scalar(df, epoch_col, base, outpath)

        outpath_diff = os.path.join(args.outdir, f"{base}_diff.png")
        plot_diff_scalar(df, epoch_col, base, outpath_diff)

    # --- Class-wise metrics
    classwise: Dict[str, Dict[int, Dict[str, str]]] = {}
    for col in df.columns:
        m = CLASS_METRIC_RE.match(col)
        if not m:
            continue
        metric, cid_str, split = m.group(1), m.group(2), m.group(3)
        cid = int(cid_str)
        classwise.setdefault(metric, {}).setdefault(cid, {})[split] = col

    for metric_name, per_class in classwise.items():
        class_ids = sorted([cid for cid, splits in per_class.items() if "train" in splits and "val" in splits])
        if not class_ids:
            continue

        outpath = os.path.join(args.outdir, f"{metric_name}_all_classes.png")
        plot_all_classes_train_val(df, epoch_col, metric_name, class_ids, outpath)

        outpath_diff = os.path.join(args.outdir, f"{metric_name}_diff_all_classes.png")
        plot_diff_classwise(df, epoch_col, metric_name, class_ids, outpath_diff)

    if args.show:
        print(f"[INFO] Saved plots to: {args.outdir}")
        print("[INFO] Open them with an image viewer.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())