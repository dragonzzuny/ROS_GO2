#!/usr/bin/env python3
"""
Render saved coverage heatmaps (*.npy) into a single PNG (grid) + per-map PNGs.

Usage:
  python scripts/render_saved_heatmaps.py \
    --dir runs/multi_map_ppo/20251229-231357/coverage/update_350
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


MAP_ORDER = [
    "map_large_square",
    "map_corridor",
    "map_l_shaped",
    "map_office_building",
    "map_warehouse",
    "map_campus",
]


def load_heatmap(path: Path) -> np.ndarray:
    a = np.load(path)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape={a.shape} in {path}")
    return a


def stats_text(hm: np.ndarray) -> str:
    total_visits = int(hm.sum())
    cells_visited = int((hm > 0).sum())
    total_cells = hm.size
    cov = 100.0 * cells_visited / max(1, total_cells)
    return f"Coverage: {cov:.1f}%\nCells: {cells_visited}/{total_cells}\nVisits: {total_visits}"


def render_one(ax, hm: np.ndarray, title: str, use_log: bool = True, vmax: float | None = None):
    img = np.log1p(hm) if use_log else hm
    im = ax.imshow(img, origin="lower", interpolation="bilinear")
    if vmax is not None:
        # vmax는 log스케일일 때는 log1p(vmax)로 맞춰줌
        im.set_clim(0, np.log1p(vmax) if use_log else vmax)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02, 0.98, stats_text(hm),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="coverage/update_xxx directory containing *_heatmap.npy")
    ap.add_argument("--out", default=None, help="output directory (default: <dir>/png)")
    ap.add_argument("--grid-name", default="coverage_heatmaps_grid.png")
    ap.add_argument("--no-log", action="store_true", help="disable log1p scale")
    ap.add_argument("--vmax", type=float, default=None, help="clip max visits for visualization (optional)")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    out_dir = Path(args.out) if args.out else (in_dir / "png")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect heatmaps
    heatmaps = {}
    for m in MAP_ORDER:
        p = in_dir / f"{m}_heatmap.npy"
        if p.exists():
            heatmaps[m] = load_heatmap(p)

    if not heatmaps:
        raise SystemExit(f"No heatmaps found in {in_dir}")

    # Per-map PNGs
    for m, hm in heatmaps.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        im = render_one(ax, hm, m, use_log=not args.no_log, vmax=args.vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1+visits)" if not args.no_log else "visits")
        fig.tight_layout()
        out_path = out_dir / f"{m}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out_path}")

    # Grid PNG (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    ims = []
    for i, m in enumerate(MAP_ORDER):
        r, c = divmod(i, 3)
        ax = axes[r][c]
        if m in heatmaps:
            im = render_one(ax, heatmaps[m], m, use_log=not args.no_log, vmax=args.vmax)
            ims.append(im)
        else:
            ax.axis("off")

    # One shared colorbar (use first im)
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("log(1+visits)" if not args.no_log else "visits")

    fig.suptitle(f"Coverage Heatmaps (from saved .npy) - {in_dir.name}", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()
    grid_path = out_dir / args.grid_name
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Grid saved: {grid_path}")


if __name__ == "__main__":
    main()
