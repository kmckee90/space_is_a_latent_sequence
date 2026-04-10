"""
vis.py
------
Transition graph visualization for a trained CSCG.

Uses igraph's Kamada-Kawai layout (C implementation, fast) and renders
the result with matplotlib.  Matches the visual style of Figure 6 in the paper.
"""

from __future__ import annotations

import numpy as np
import igraph as ig
import matplotlib.pyplot as plt


def plot_transition_graph(
    T: np.ndarray,
    ax: plt.Axes = None,
    threshold: float = 0.01,
    node_size: int = 100,
    node_color: str = "burlywood",
    edge_color: str = "gray",
    title: str = None,
) -> plt.Axes:
    """
    Draw a directed transition graph using igraph's Kamada-Kawai layout.

    Parameters
    ----------
    T          : (n, n) row-stochastic transition matrix
    ax         : matplotlib Axes (created if None)
    threshold  : only draw edges with probability > threshold
    node_size  : scatter marker size for state nodes
    node_color : border colour of the nodes
    edge_color : colour of the directed arrows
    title      : optional axes title

    Returns
    -------
    ax : the matplotlib Axes with the graph drawn
    """
    n = T.shape[0]

    edges = [(i, j) for i in range(n) for j in range(n) if T[i, j] > threshold]

    g = ig.Graph(n=n, edges=edges, directed=False)
    layout = g.layout_kamada_kawai()
    coords = np.array(layout.coords, dtype=float)   # (n, 2)

    # Normalise to [0, 1]²
    if n > 1:
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        span = hi - lo
        span[span == 0] = 1.0
        coords = (coords - lo) / span

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.set_aspect("equal")
    ax.axis("off")

    for src, dst in edges:
        ax.annotate(
            "",
            xy=coords[dst], xytext=coords[src],
            arrowprops=dict(
                arrowstyle="-|>",
                color=edge_color,
                alpha=0.5,
                lw=0.8,
                connectionstyle="arc3,rad=0.1",
                shrinkA=6,
                shrinkB=6,
            ),
            zorder=1,
        )

    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=node_size,
        facecolors="white",
        edgecolors=node_color,
        linewidths=1.5,
        zorder=2,
    )

    if title:
        ax.set_title(title, fontsize=12)

    return ax
