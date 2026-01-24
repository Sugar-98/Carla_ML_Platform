"""
Plot training log CSV.

Requirements (from user):
- Plot Train and Val together in the same figure for each metric.
- For class-wise metrics, plot all classes together in a single figure (per metric type).
- Adjust legend to be clean and readable.

Usage:
  python plot_log_csv.py --csv log.csv --outdir plots --show

Outputs:
  plots/loss.png
  plots/time_train_val.png
  plots/mIoU.png
  plots/m_precision.png
  ...
  plots/precision_all_classes.png
  plots/recall_all_classes.png
  plots/f1_all_classes.png
  plots/iou_all_classes.png
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CLASS_METRIC_RE = re.compile(r"^(precision|recall|f1|iou)_(\d+)_(train|val)$")
SPLIT_SUFFIX_RE = re.compile(r"^(.*)_(train|val)$")


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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

    fig = plt.figure(figsize=(9.5, 5.5))
    ax = fig.add_subplot(111)

    ax.plot(x, y_train, label="Train")
    ax.plot(x, y_val, label="Val")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_base)
    ax.set_title(title or metric_base)

    ax.grid(True, alpha=0.3)

    # Clean legend: inside, best spot
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


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


def plot_all_classes_train_val(
    df: pd.DataFrame,
    epoch_col: str,
    metric_name: str,
    class_ids: List[int],
    outpath: str,
) -> None:
    """
    One figure per metric_name (precision/recall/f1/iou), all classes together.
    Train+Val must be on same figure.
    Use same color per class, different linestyle for train/val.
    """
    x = df[epoch_col].to_numpy()

    # Use a colormap to keep class colors stable (avoid relying on default cycle).
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

        # Same color for class, linestyle differs by split
        ax.plot(x, y_train, linestyle="-", color=colors[idx], label=f"C{c} Train")
        ax.plot(x, y_val, linestyle="--", color=colors[idx], label=f"C{c} Val")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}: all classes (Train solid / Val dashed)")
    ax.grid(True, alpha=0.3)

    # Legend outside to avoid clutter; multi-column
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
            columnspacing=1.0,
            handlelength=2.4,
            borderaxespad=0.0,
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


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

    # Ensure numeric epoch
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).reset_index(drop=True)
    epoch_col = "epoch"

    # --- 1) Handle special paired metrics that aren't *_train/_val bases (if any)
    # In your CSV, time columns are time_train(h), time_val(h) (not time_train)
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

    # --- 2) Scalar metrics with *_train and *_val
    # Detect bases from columns ending with _train/_val
    bases: Dict[str, set] = {}
    for col in df.columns:
        m = SPLIT_SUFFIX_RE.match(col)
        if not m:
            continue
        base, split = m.group(1), m.group(2)
        bases.setdefault(base, set()).add(split)

    # We will plot any base that has both train and val,
    # but we will EXCLUDE class-wise bases like precision_0, recall_3, etc (handled later).
    def is_classwise_base(base: str) -> bool:
        # base like "precision_0" / "iou_10" etc.
        return bool(re.match(r"^(precision|recall|f1|iou)_\d+$", base))

    scalar_bases = sorted([b for b, splits in bases.items() if splits == {"train", "val"} and not is_classwise_base(b)])

    for base in scalar_bases:
        outpath = os.path.join(args.outdir, f"{base}.png")
        plot_train_val_scalar(df, epoch_col, base, outpath)

    # --- 3) Class-wise metrics: precision/recall/f1/iou all classes in one fig each
    classwise: Dict[str, Dict[int, Dict[str, str]]] = {}  # metric -> class_id -> split -> colname
    for col in df.columns:
        m = CLASS_METRIC_RE.match(col)
        if not m:
            continue
        metric, cid_str, split = m.group(1), m.group(2), m.group(3)
        cid = int(cid_str)
        classwise.setdefault(metric, {}).setdefault(cid, {})[split] = col

    for metric_name, per_class in classwise.items():
        # Keep classes that have both train and val
        class_ids = sorted([cid for cid, splits in per_class.items() if "train" in splits and "val" in splits])
        if not class_ids:
            continue
        outpath = os.path.join(args.outdir, f"{metric_name}_all_classes.png")
        plot_all_classes_train_val(df, epoch_col, metric_name, class_ids, outpath)

    if args.show:
        # Re-open saved figures quickly (optional): simplest is to just notify;
        # interactive re-open is environment-specific, so we do not auto-open files here.
        print(f"[INFO] Saved plots to: {args.outdir}")
        print("[INFO] Use an image viewer, or re-run and customize to plt.show() per plot if needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())