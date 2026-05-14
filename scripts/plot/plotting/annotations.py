from __future__ import annotations

from textwrap import fill


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        ha='left',
        va='bottom',
        color='#1E293B',
    )


def add_header_block(ax, *, title: str, subtitle: str, takeaway: str) -> None:
    ax.axis('off')
    # Add a subtle background for the header
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='#F8FAFC', edgecolor='#E2E8F0', linewidth=0.8, zorder=0))
    
    ax.text(
        0.015,
        0.85,
        title,
        fontsize=14,
        fontweight='bold',
        ha='left',
        va='top',
        color='#0F172A',
        zorder=1
    )
    ax.text(
        0.015,
        0.45,
        fill(subtitle, width=105),
        fontsize=8.5,
        ha='left',
        va='top',
        color='#334155',
        zorder=1
    )
    ax.text(
        0.985,
        0.45,
        f'Key Insights: {takeaway}',
        fontsize=8.2,
        fontweight='bold',
        ha='right',
        va='top',
        color='#475569',
        zorder=1
    )


def add_caption_note(ax, text: str) -> None:
    ax.text(
        0.0,
        -0.15,
        fill(text, width=65),
        transform=ax.transAxes,
        fontsize=7.0,
        ha='left',
        va='top',
        color='#64748B',
        fontstyle='italic'
    )


def add_callout(ax, x: float, y: float, text: str, *, ha: str = 'left') -> None:
    ax.text(
        x,
        y,
        fill(text, width=28),
        fontsize=7.5,
        ha=ha,
        va='center',
        color='#0F172A',
        bbox={
            'boxstyle': 'round,pad=0.3,rounding_size=0.2',
            'facecolor': '#FFFFFF',
            'edgecolor': '#94A3B8',
            'linewidth': 0.8,
            'alpha': 0.95
        },
        zorder=6,
    )


def add_zone_label(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=7.2,
        fontweight='bold',
        ha='center',
        va='center',
        color='#475569',
        bbox={
            'boxstyle': 'round,pad=0.25,rounding_size=0.2',
            'facecolor': '#FFFFFF',
            'edgecolor': '#CBD5E1',
            'linewidth': 0.7,
            'alpha': 0.9
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
    ha: str = 'left',
    fontsize: float = 7.0,
    facecolor: str = '#FFFFFF',
    edgecolor: str = '#CBD5E1',
    textcolor: str = '#475569',
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=transform if transform is not None else ax.transAxes,
        fontsize=fontsize,
        fontweight='bold',
        ha=ha,
        va='center',
        color=textcolor,
        bbox={
            'boxstyle': 'round,pad=0.3,rounding_size=0.2',
            'facecolor': facecolor,
            'edgecolor': edgecolor,
            'linewidth': 0.8,
        },
        zorder=6,
    )

import matplotlib.pyplot as plt
