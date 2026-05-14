from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np

from scripts.plot.common import CLAIM_COLORS, PATHWAY_COLORS, CONFIDENCE_COLORS, NECESSITY_COLORS

ULTRA_PREMIUM_FONT_SIZE = 8.5
ULTRA_PREMIUM_FONT_FAMILY = ['DejaVu Sans']
ULTRA_PREMIUM_RC = {
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FAFBFC',
    'savefig.facecolor': '#FFFFFF',
    'font.family': ULTRA_PREMIUM_FONT_FAMILY,
    'font.size': ULTRA_PREMIUM_FONT_SIZE,
    'axes.titlesize': 11.0,
    'axes.titleweight': 'semibold',
    'axes.titlepad': 10.0,
    'axes.labelsize': 9.5,
    'xtick.labelsize': ULTRA_PREMIUM_FONT_SIZE,
    'ytick.labelsize': ULTRA_PREMIUM_FONT_SIZE,
    'legend.fontsize': ULTRA_PREMIUM_FONT_SIZE,
    'axes.edgecolor': '#1E293B',
    'axes.linewidth': 1.0,
    'grid.color': '#E2E8F0',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def configure_publication_theme():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:  # optional styling package
        import scienceplots  # noqa: F401

        plt.style.use(['science', 'nature', 'no-latex'])
    except ModuleNotFoundError:
        plt.style.use('default')
    sns.set_theme(style='white')
    plt.rcParams.update(ULTRA_PREMIUM_RC)
    return plt

def pathway_color(pathway: str) -> str:
    return PATHWAY_COLORS.get(str(pathway).lower(), '#64748B')

def confidence_color(tier: str) -> str:
    return CONFIDENCE_COLORS.get(str(tier).lower(), '#64748B')

def soften_hex(hex_color: str, weight: float = 0.85) -> str:
    try:
        color = hex_color.lstrip('#')
        rgb = [int(color[i:i+2], 16) for i in (0, 2, 4)]
        mixed = [int(c * (1.0 - weight) + 255 * weight) for c in rgb]
        return '#{:02X}{:02X}{:02X}'.format(*mixed)
    except:
        return '#F1F5F9'

def style_axis(ax, *, grid_axis: str = 'y', grid: bool = True):
    if grid:
        ax.grid(True, axis=grid_axis, color='#E2E8F0', linewidth=0.6, linestyle='--', alpha=0.7)
    else:
        ax.grid(False)
    ax.tick_params(length=0, pad=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['bottom'].set_color('#94A3B8')

def draw_gradient_barh(ax, y, width, height, color_start, **kwargs):
    color_end = soften_hex(color_start, weight=0.35)
    n_segments = 50
    zorder = kwargs.get('zorder', 3)
    for i in range(n_segments):
        frac = i / n_segments
        color = mcolors.to_hex(mcolors.to_rgb(color_start) + (np.array(mcolors.to_rgb(color_end)) - np.array(mcolors.to_rgb(color_start))) * frac)
        ax.barh(y, width / n_segments, left=width * frac, height=height, color=color, edgecolor='none', zorder=zorder)
    ax.barh(y + height*0.4, width, height=height*0.08, color='#FFFFFF', alpha=0.2, edgecolor='none', zorder=zorder+0.1)

def add_status_badge(ax, x, y, text, color, **kwargs):
    transform = kwargs.get('transform', ax.transAxes)
    ax.text(x, y, text, transform=transform, fontsize=8, ha='center', va='center', color='#FFFFFF', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5,rounding_size=0.3', facecolor=color, edgecolor='none', alpha=0.9),
            zorder=10)

def add_innovation_glow(ax, x, y, color, s=150, alpha=0.1):
    for i in range(1, 6):
        ax.scatter(x, y, s=s*(1+i*0.5), color=color, alpha=alpha/i, edgecolors='none', zorder=1)

def claim_color(group: str) -> str:
    return CLAIM_COLORS.get(str(group).lower(), '#64748B')

def scenario_marker(scenario_name: str) -> str:
    markers = {'baseline_region_case': 'o', 'high_supply_case': 's', 'policy_support_case': 'D'}
    return markers.get(str(scenario_name), 'o')
