#!/usr/bin/env python3
"""Parse a weaver training log and plot loss/accuracy + ROC AUC matrix.

Supports both single-training logs and k-fold cross-validation logs.
Class labels and number of classes are inferred from the log itself.
"""

import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = (
    "logs/ttH_0L_ParT_20260523-160945_ParT_ranger_lr0.004_batch2048"
    "_laurids_ul17v15_split_njet6.log"
)

# Fallback used only if the log contains no _label_ definition
_FALLBACK_LABELS = [
    "ttHcc", "ttHbb", "ttZqq", "ttZcc", "ttZbb",
    "ttLF", "ttcj", "ttcc", "ttbj", "ttbb",
    "zbb", "zcc", "zqq", "wcq", "wqq", "qcd",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_class_labels(lines):
    """Extract ordered class names from the _label_ np.argmax(np.stack(...)) line."""
    re_label = re.compile(r"'_label_'.*?np\.argmax\(np\.stack\(\[(.*?)\]")
    for line in lines:
        m = re_label.search(line)
        if m:
            names = re.findall(r"ak\.to_numpy\((\w+)\)", m.group(1))
            if names:
                return names
    return _FALLBACK_LABELS


def _isqrt(n):
    """Integer square root; returns None if n is not a perfect square."""
    r = int(math.isqrt(n))
    return r if r * r == n else None


# ── Low-level parsing of a single fold's lines ──────────────────────────────

def _parse_fold_lines(lines, class_labels):
    """Return (train_loss, train_acc, val_metric, roc_scores, roc_matrices)."""
    train_loss, train_acc, val_metric, roc_scores = [], [], [], []
    roc_matrices = {}  # epoch -> np.ndarray

    re_train = re.compile(
        r"Train AvgLoss:\s*([\d.]+),\s*AvgAcc:\s*([\d.]+)"
    )
    re_val = re.compile(
        r"Epoch #(\d+): Current validation metric:\s*([\d.]+)"
    )
    re_validating = re.compile(r"Epoch #(\d+) validating")
    re_roc_scalar = re.compile(r"^\s*- roc_auc_score:\s*$")
    re_roc_matrix = re.compile(r"^\s*- roc_auc_score_matrix:\s*$")
    re_info = re.compile(r"^\[.*\] INFO:")

    n_classes = len(class_labels)
    current_epoch = -1
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        m = re_validating.search(line)
        if m:
            current_epoch = int(m.group(1))
            i += 1
            continue

        m = re_train.search(line)
        if m:
            train_loss.append(float(m.group(1)))
            train_acc.append(float(m.group(2)))
            i += 1
            continue

        m = re_val.search(line)
        if m:
            val_metric.append((int(m.group(1)), float(m.group(2))))
            i += 1
            continue

        if re_roc_scalar.search(line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                try:
                    roc_scores.append(float(lines[j].strip()))
                except ValueError:
                    pass
                i = j + 1
            else:
                i += 1
            continue

        if re_roc_matrix.search(line):
            chunks = []
            j = i + 1
            while j < len(lines):
                ln = lines[j].rstrip()
                if re_info.match(ln):
                    break
                if ln.strip():
                    chunks.append(ln.strip())
                if ln.endswith("]]"):
                    j += 1
                    break
                j += 1
            raw = " ".join(chunks)
            raw = raw.replace("[[", "[").replace("]]", "]")
            raw = re.sub(r"\]\s*\[", " ", raw).strip("[]")
            try:
                vals = [float(x) for x in raw.split()]
                side = _isqrt(len(vals))
                if side is not None and side == n_classes:
                    roc_matrices[current_epoch] = (
                        np.array(vals).reshape(side, side)
                    )
            except Exception:
                pass
            i = j
            continue

        i += 1

    return train_loss, train_acc, val_metric, roc_scores, roc_matrices


# ── Top-level log parser (handles single runs and k-fold) ───────────────────

def parse_log(path):
    """Return a list of fold dicts.  Single-training logs yield one element.

    Each dict has keys:
        fold, n_folds, class_labels,
        train_loss, train_acc, val_metric, roc_scores, roc_matrices
    """
    with open(path) as fh:
        lines = fh.readlines()

    re_fold = re.compile(
        r"=== Running cross validation, fold (\d+) of (\d+) ==="
    )

    boundaries = []
    for idx, line in enumerate(lines):
        m = re_fold.match(line.strip())
        if m:
            boundaries.append((idx, int(m.group(1)), int(m.group(2))))

    if not boundaries:
        labels = _parse_class_labels(lines)
        data = _parse_fold_lines(lines, labels)
        return [dict(
            fold=None,
            n_folds=None,
            class_labels=labels,
            train_loss=data[0],
            train_acc=data[1],
            val_metric=data[2],
            roc_scores=data[3],
            roc_matrices=data[4],
        )]

    folds = []
    for i, (start, fold_idx, n_folds) in enumerate(boundaries):
        end = (
            boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        )
        fold_lines = lines[start:end]
        labels = _parse_class_labels(fold_lines)
        data = _parse_fold_lines(fold_lines, labels)
        folds.append(dict(
            fold=fold_idx,
            n_folds=n_folds,
            class_labels=labels,
            train_loss=data[0],
            train_acc=data[1],
            val_metric=data[2],
            roc_scores=data[3],
            roc_matrices=data[4],
        ))
    return folds


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_curves(folds, outdir):
    is_cv = folds[0]["fold"] is not None
    n_folds = folds[0]["n_folds"] if is_cv else 1
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_folds, 1)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    title_suffix = (
        f"k={n_folds} cross-validation" if is_cv else "single training"
    )
    fig.suptitle(f"Training curves — ParT ({title_suffix})", fontsize=13)

    for fold_dict, color in zip(folds, colors):
        label = f"Fold {fold_dict['fold']}" if is_cv else None
        epochs = list(range(len(fold_dict["train_loss"])))
        best = (
            int(np.argmax(fold_dict["roc_scores"]))
            if fold_dict["roc_scores"] else 0
        )
        kw_line = dict(color=color, label=label)
        kw_best = dict(color=color, ls="--", lw=0.8, alpha=0.6)

        axes[0].plot(epochs, fold_dict["train_loss"], **kw_line)
        axes[0].axvline(best, **kw_best)
        axes[1].plot(epochs, fold_dict["train_acc"], **kw_line)
        axes[1].axvline(best, **kw_best)
        axes[2].plot(epochs, fold_dict["roc_scores"], **kw_line)
        axes[2].axvline(best, **kw_best)

    for ax, (ylabel, title) in zip(axes, [
        ("Cross-entropy loss", "Training loss"),
        ("Accuracy",           "Training accuracy"),
        ("ROC AUC score",      "Validation ROC AUC"),
    ]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if is_cv:
            ax.legend(fontsize=8)

    fig.tight_layout()
    # out = os.path.join(outdir, "training_curves.pdf")
    out = os.path.join(outdir, "training_curves.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_roc_matrix(mat, epoch, fold, class_labels, outdir):
    n = len(class_labels)
    full = mat.copy()
    for row in range(n):
        for col in range(row):
            full[row, col] = mat[col, row]
    np.fill_diagonal(full, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(full, vmin=0.5, vmax=1.0, cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pairwise ROC AUC")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)

    fold_str = f", fold {fold}" if fold is not None else ""
    ax.set_title(
        f"Pairwise ROC AUC matrix — best epoch {epoch}{fold_str}",
        fontsize=12,
    )

    fontsize = max(4, 7 - n // 4)
    for row in range(n):
        for col in range(n):
            if row == col:
                continue
            val = full[row, col]
            color = "black" if 0.65 < val < 0.9 else "white"
            ax.text(
                col, row, f"{val:.3f}",
                ha="center", va="center", fontsize=fontsize, color=color,
            )

    fig.tight_layout()
    fold_tag = f"_fold{fold}" if fold is not None else ""
    out = os.path.join(
        outdir, f"roc_auc_matrix_epoch{epoch}{fold_tag}.png"
    )
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves and ROC AUC matrix from a "
                    "weaver training log (supports k-fold CV)."
    )
    parser.add_argument(
        "infile",
        nargs="?",
        default=LOG_FILE,
        help="Path to the weaver training log (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--outdir",
        default="logs",
        help="Output directory for the PDF plots (default: %(default)s)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Parsing {args.infile} …")
    folds = parse_log(args.infile)

    is_cv = folds[0]["fold"] is not None
    print(
        f"  Mode:  {'cross-validation' if is_cv else 'single training'}"
        + (f"  ({len(folds)} folds)" if is_cv else "")
    )
    print(f"  Classes ({len(folds[0]['class_labels'])}):"
          f" {folds[0]['class_labels']}")

    for fd in folds:
        tag = f"fold {fd['fold']}" if is_cv else "run"
        best = int(np.argmax(fd["roc_scores"])) if fd["roc_scores"] else -1
        auc_str = (
            f" (ROC AUC = {fd['roc_scores'][best]:.5f})" if best >= 0 else ""
        )
        mat_count = len(fd["roc_matrices"])
        print(
            f"  {tag:>8}: {len(fd['train_loss'])} epochs,"
            f" best epoch {best}{auc_str},"
            f" {mat_count} matrices"
        )

    plot_curves(folds, args.outdir)

    for fd in folds:
        if not fd["roc_scores"]:
            continue
        best = int(np.argmax(fd["roc_scores"]))
        mats = fd["roc_matrices"]
        labels = fd["class_labels"]
        if best in mats:
            plot_roc_matrix(mats[best], best, fd["fold"], labels, args.outdir)
        else:
            available = sorted(mats)
            if available:
                fallback = available[-1]
                fold_tag = f"fold {fd['fold']} " if is_cv else ""
                print(
                    f"  {fold_tag}matrix for epoch {best} not found;"
                    f" using epoch {fallback}."
                )
                plot_roc_matrix(
                    mats[fallback], fallback, fd["fold"], labels, args.outdir
                )


if __name__ == "__main__":
    main()
