from __future__ import annotations

from textwrap import fill


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_header_block(ax, *, title: str, subtitle: str, takeaway: str) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        0.98,
        title,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        color="#0F172A",
    )
    ax.text(
        0.0,
        0.50,
        fill(subtitle, width=110),
        fontsize=7.8,
        ha="left",
        va="top",
        color="#334155",
    )
    ax.text(0.995, 0.50, takeaway, fontsize=7.5, ha="right", va="top", color="#475569")


def add_caption_note(ax, text: str) -> None:
    ax.text(
        0.01,
        -0.13,
        fill(text, width=62),
        transform=ax.transAxes,
        fontsize=6.5,
        ha="left",
        va="top",
        color="#475569",
    )


def add_callout(ax, x: float, y: float, text: str, *, ha: str = "left") -> None:
    ax.text(
        x,
        y,
        fill(text, width=26),
        fontsize=7.0,
        ha=ha,
        va="center",
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.18",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "linewidth": 0.7,
        },
        zorder=6,
    )


def add_zone_label(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=6.7,
        ha="center",
        va="center",
        color="#475569",
        bbox={
            "boxstyle": "round,pad=0.20,rounding_size=0.16",
            "facecolor": "white",
            "edgecolor": "#E2E8F0",
            "linewidth": 0.6,
        },
        zorder=5,
    )


def add_badge(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    transform=None,
    ha: str = "left",
    fontsize: float = 6.5,
    facecolor: str = "white",
    edgecolor: str = "#D9E2EC",
    textcolor: str = "#475569",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=transform if transform is not None else ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va="center",
        color=textcolor,
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.18",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 0.7,
        },
        zorder=6,
    )
