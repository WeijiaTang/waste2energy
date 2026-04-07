from __future__ import annotations

from matplotlib.gridspec import GridSpec


def create_main_figure(plt):
    fig = plt.figure(figsize=(11.3, 5.6))
    grid = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.28, 1.0, 0.95],
        height_ratios=[0.19, 1.0],
        wspace=0.28,
        hspace=0.18,
    )
    header_ax = fig.add_subplot(grid[0, :])
    score_ax = fig.add_subplot(grid[1, 0])
    tradeoff_ax = fig.add_subplot(grid[1, 1])
    evidence_ax = fig.add_subplot(grid[1, 2])
    return fig, {
        "header": header_ax,
        "score": score_ax,
        "tradeoff": tradeoff_ax,
        "evidence": evidence_ax,
    }


def create_supporting_figure(plt):
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.2))
    return fig, ax


def create_three_panel_supporting_figure(plt, *, figsize=(10.6, 3.8)):
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    return fig, axes


def create_three_panel_polar_figure(plt, *, figsize=(11.0, 4.1)):
    fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw={"projection": "polar"})
    return fig, axes
