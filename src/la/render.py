"""Render a plot spec to a PNG.

Deterministic on purpose: the same spec always produces the same figure, so a
figure can be compared by the numbers in its spec rather than by pixels. The
renderer never guesses. If a required key is missing or malformed it raises,
and the caller is expected to have run `validate_spec` first.

Matplotlib's Agg backend is selected at import time so this works headless.
"""
from __future__ import annotations

import math
import os

import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# An equal-aspect figure and a two-panel heatmap both make tight_layout
# complain. The complaints are noise here - the layout is deliberate - so
# they are silenced rather than left to flood a corpus build.
warnings.filterwarnings("ignore", message="Tight layout not applied")
warnings.filterwarnings("ignore", message="Ignoring fixed .* limits")
# that one is a log record from matplotlib, not a warning
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# A single restrained palette. The book's aesthetic does not need to be
# reproduced here, but the figures do need to be legible at print size.
INK = "#141A26"
SLATE = "#4A566B"
GRID = "#E4E9F0"
ACCENT = "#C1552F"
ACCENT2 = "#2B4470"
ACCENT3 = "#2E7D7B"
BAND = "#DFA63E"

FIGSIZE = (4.6, 3.4)
DPI = 200


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SLATE, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#C9D2DE")


def _arrow(ax, xy, xytext, color, label=None, label_offset=(4, 6)):
    """An arrow from `xytext` to `xy`, labelled at its head.

    The label goes at the head deliberately: several arrows in one figure
    share the origin as their tail, so labelling at the tail stacks every
    label on the same point.
    """
    ax.annotate("", xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text(xy[0] + label_offset[0], xy[1] + label_offset[1],
                label, color=color, fontsize=9, va="center")


def _annotate(ax, spec):
    """Title and annotations, placed in FIGURE coordinates.

    Axes3D.text needs a third coordinate, so an axes-relative annotation
    cannot be written for a 3D figure. Figure coordinates work for both.
    The title sits at the very top and the annotations below it, so the two
    do not collide.
    """
    labels = spec.get("labels", {}) or {}
    title = labels.get("title")
    top = 0.972
    if title:
        ax.figure.text(0.03, top, title, fontsize=9.5, fontweight="bold",
                       color=INK, va="top", ha="left")
        top -= 0.055
    for i, s in enumerate(spec.get("annotate", []) or []):
        ax.figure.text(0.03, top - 0.038 * i, s, fontsize=9, color=INK,
                       va="top", ha="left")


def render(spec: dict, out_path: str) -> str:
    """Render `spec` to `out_path` (PNG) and return the path."""
    kind = spec["kind"]
    data = spec.get("data", {}) or {}
    labels = spec.get("labels", {}) or {}

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("#FBF7EF")
    ax.set_facecolor("#FBF7EF")

    handler = _HANDLERS[kind]
    # A handler may build its own axes (the two-panel row-echelon figure does).
    # It returns the axes it wants decorated, or None if it did its own layout.
    target = handler(fig, ax, data, labels)
    if target is not None:
        _style(target)
        _annotate(target, spec)
        fig.tight_layout()
    else:
        try:
            fig.tight_layout()
        except ValueError:
            pass
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# --------------------------------------------------------------- handlers --

def _limits_2d(ax, points, pad=0.22, force_equal=True):
    """Set axis limits explicitly.

    `ax.annotate` does not participate in autoscaling, so any figure drawn
    with annotations would otherwise keep matplotlib's default 0..1 limits and
    render its arrows outside the visible area.
    """
    pts = [tuple(p) for p in points if p is not None and len(p) == 2]
    if not pts:
        return
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span = max(span_x, span_y, 1e-9)
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    half = span * (0.5 + pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    if force_equal:
        ax.set_aspect("equal", adjustable="box")


def _h_basis_2d(fig, ax, data, labels):
    vecs = np.asarray(data["vectors"], dtype=float)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for i, v in enumerate(vecs):
        c = [ACCENT, ACCENT2, ACCENT3, BAND][i % 4]
        _arrow(ax, (0, 0), tuple(v), c, labels.get(f"v{i}"))
    ax.set_xlabel(labels.get("x", "x"))
    ax.set_ylabel(labels.get("y", "y"))
    _limits_2d(ax, list(vecs) + [(0, 0)])
    return ax


def _h_basis_3d(fig, ax, data, labels):
    """A three-dimensional basis.

    The default axes is two-dimensional, so it is replaced with a 3D
    projection; the projection has to be registered with the figure or the
    scatter call below is the two-dimensional one and rejects the third
    coordinate.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)
    ax.remove()
    ax = fig.add_subplot(111, projection="3d")
    vecs = np.asarray(data["vectors"], dtype=float)
    for i, v in enumerate(vecs):
        c = [ACCENT, ACCENT2, ACCENT3, BAND][i % 4]
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, lw=2.2)
        ax.scatter([v[0]], [v[1]], [v[2]], color=c, s=34, zorder=5)
        if labels.get(f"v{i}"):
            ax.text(v[0], v[1], v[2], "  " + labels[f"v{i}"], color=c, fontsize=9)
    lim = max(1.0, float(np.abs(vecs).max()) if vecs.size else 1.0) * 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel(labels.get("x", "x"))
    ax.set_ylabel(labels.get("y", "y"))
    ax.set_zlabel(labels.get("z", "z"))
    ax.view_init(elev=22, azim=-58)
    return ax


def _h_eigenvalue_spectrum(fig, ax, data, labels):
    vals = data["values"]
    pts = []
    for v in vals:
        if isinstance(v, (list, tuple)):
            pts.append((float(v[0]), float(v[1])))
        else:
            pts.append((float(v), 0.0))
    ax.axhline(0, color=INK, lw=0.9)
    ax.axvline(0, color="#C9D2DE", lw=0.9, ls="--")
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.scatter(xs, ys, s=64, color=ACCENT, zorder=5, edgecolor="#FBF7EF", linewidth=1.2)
    for (x, y), lbl in zip(pts, labels.get("points", []) or []):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(7, 6),
                    fontsize=9, color=INK)
    if xs:
        m = max(1.0, max(abs(min(xs)), abs(max(xs))) * 1.35)
        ax.set_xlim(-m, m)
    if ys:
        m2 = max(0.6, max(abs(min(ys)), abs(max(ys))) * 1.5)
        ax.set_ylim(-m2, m2)
    ax.set_xlabel(labels.get("x", "Re(λ)"))
    ax.set_ylabel(labels.get("y", "Im(λ)"))
    ax.set_aspect("equal", adjustable="box")
    return ax


def _h_row_echelon(fig, ax, data, labels):
    """Two panels: the original matrix and its reduced row echelon form."""
    m = np.asarray(data["matrix"], dtype=float)
    e = np.asarray(data["echelon"], dtype=float)
    ax.remove()
    axes = fig.subplots(1, 2)
    for a, mat, title in ((axes[0], m, labels.get("left", "A")),
                          (axes[1], e, labels.get("right", "rref(A)"))):
        a.set_facecolor("#FBF7EF")
        _heatmap(a, mat)
        a.set_title(title, fontsize=9, color=INK, loc="left")
        a.tick_params(colors=SLATE, labelsize=7)
        for s in a.spines.values():
            s.set_visible(False)
    for a in axes:
        a.text(0.0, -0.22, "", transform=a.transAxes)
    return None


def _heatmap(ax, mat):
    m = np.asarray(mat, dtype=float)
    if m.size == 0:
        return
    im = ax.imshow(m, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(m.shape[1]))
    ax.set_yticks(range(m.shape[0]))
    ax.set_xticklabels([str(i) for i in range(m.shape[1])], fontsize=7)
    ax.set_yticklabels([str(i) for i in range(m.shape[0])], fontsize=7)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=8,
                    color="#FFFFFF" if abs(v) > 0.55 * (np.abs(m).max() or 1) else INK)
    fig = ax.get_figure()
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7, colors=SLATE)
    cb.outline.set_visible(False)


def _h_matrix_heatmap(fig, ax, data, labels):
    _heatmap(ax, np.asarray(data["matrix"], dtype=float))
    return ax


def _h_subspace_2d(fig, ax, data, labels):
    o = data.get("origin", [0, 0])
    vecs = np.asarray(data["vectors"], dtype=float)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for i, v in enumerate(vecs):
        c = [ACCENT, ACCENT2, ACCENT3][i % 3]
        _arrow(ax, tuple(o), (o[0] + v[0], o[1] + v[1]), c, labels.get(f"v{i}"))
    ax.set_xlabel(labels.get("x", "x"))
    ax.set_ylabel(labels.get("y", "y"))
    _limits_2d(ax, list(vecs) + [tuple(o)])
    return ax


def _h_projection(fig, ax, data, labels):
    v = data["vector"]; p = data["projection"]; r = data["residual"]; a = data["axis"]
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    _arrow(ax, (0, 0), tuple(a), "#C9D2DE", labels.get("axis", "axis"))
    _arrow(ax, (0, 0), tuple(p), ACCENT2, labels.get("projection", "proj"))
    _arrow(ax, tuple(p), tuple(v), ACCENT, labels.get("residual", "residual"))
    ax.set_xlabel(labels.get("x", "x"))
    ax.set_ylabel(labels.get("y", "y"))
    _limits_2d(ax, [v, p, r, a, (0, 0)])
    return ax


def _h_least_squares(fig, ax, data, labels):
    pts = np.asarray(data["points"], dtype=float)
    fit = np.asarray(data["fit"], dtype=float)
    ax.scatter(pts[:, 0], pts[:, 1], s=42, color=ACCENT, zorder=5,
               edgecolor="#FBF7EF", linewidth=1.0)
    if fit.size:
        ax.plot(fit[:, 0], fit[:, 1], color=ACCENT2, lw=2.2)
    ax.set_xlabel(labels.get("x", "x"))
    ax.set_ylabel(labels.get("y", "y"))
    return ax


def _h_singular_spectrum(fig, ax, data, labels):
    vals = [float(v) for v in data["values"]]
    idx = np.arange(len(vals))
    ax.bar(idx, vals, 0.55, color=ACCENT2, edgecolor="#FBF7EF", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels([labels.get(f"s{i}", str(i + 1)) for i in range(len(vals))], fontsize=8)
    ax.set_ylabel(labels.get("y", "σᵢ"))
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1.0)
    return ax


def _h_transform_unit_square(fig, ax, data, labels):
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    img = np.asarray(data["image"], dtype=float)
    ax.add_patch(Polygon(unit, closed=True, facecolor="#E4E9F0", edgecolor=SLATE,
                         lw=1.4, zorder=2))
    if img.size:
        ax.add_patch(Polygon(img, closed=True, facecolor=ACCENT, edgecolor="#9E3E20",
                             lw=1.6, alpha=0.72, zorder=3))
        for i, p in enumerate(img):
            ax.text(p[0], p[1], f"  {i}", color=INK, fontsize=9)
    ax.axhline(0, color="#C9D2DE", lw=0.8)
    ax.axvline(0, color="#C9D2DE", lw=0.8)
    ax.set_xlabel(labels.get("x", "x₁"))
    ax.set_ylabel(labels.get("y", "x₂"))
    pts = list(unit) + (list(img) if img.size else [])
    _limits_2d(ax, pts)
    # the title is drawn once, at figure level, by _annotate
    return ax


_HANDLERS = {
    "basis_2d": _h_basis_2d,
    "basis_3d": _h_basis_3d,
    "eigenvalue_spectrum": _h_eigenvalue_spectrum,
    "row_echelon": _h_row_echelon,
    "subspace_2d": _h_subspace_2d,
    "projection": _h_projection,
    "least_squares": _h_least_squares,
    "singular_spectrum": _h_singular_spectrum,
    "matrix_heatmap": _h_matrix_heatmap,
    "transform_unit_square": _h_transform_unit_square,
}


def render_many(specs: list[dict], out_dir: str, prefix: str = "fig") -> list[str]:
    paths = []
    for i, spec in enumerate(specs):
        paths.append(render(spec, os.path.join(out_dir, f"{prefix}_{i:04d}.png")))
    return paths
