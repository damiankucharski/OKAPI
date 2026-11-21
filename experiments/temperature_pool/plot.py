"""Plot temperature-minus-CLR paired differences from the published bank summary."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
FAMILIES = ('multiclass_committee', 'saturated', 'heterogeneous')
NAMES = ('Multiclass committee', 'Saturated', 'Heterogeneous experts')
STYLES = {128: ('#165b79', 'o', .13), 512: ('#b34d29', 'D', -.13)}


def render(source, output):
    data = pd.read_csv(source)
    points = data.loc[
        (data.reference == 'stack_clr') & (data.metric == 'conditional_excess')
        & data.family.isin(FAMILIES) & data.split.isin(('iid', 'shift'))
    ].set_index(['family', 'split', 'nfit'])
    expected = pd.MultiIndex.from_product([FAMILIES, ('iid', 'shift'), STYLES], names=points.index.names)
    if not points.index.is_unique or set(points.index) != set(expected) or not points.banks.eq(16).all():
        raise ValueError('Expected one paired 16-bank estimate per scenario, split and fit size')
    if not (np.isfinite(points[['mean', 'lo95', 'hi95']]).all().all() and (points.lo95 <= points['mean']).all()
            and (points['mean'] <= points.hi95).all()):
        raise ValueError('Invalid confidence interval')

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.19, right=.98, top=.71, bottom=.23, wspace=.12)
    extent = points.hi95.max() - points.lo95.min()
    limits = (points.lo95.min() - .05 * extent, points.hi95.max() + .14 * extent)
    ink, secondary = '#182e38', '#495b61'
    for ax, split, title in zip(axes, ('iid', 'shift'), ('Fresh IID cases', 'After distribution shift'), strict=True):
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#dce2e4', linewidth=.65)
        ax.axvline(0., color='#60747c', linewidth=1.)
        for boundary in (.5, 1.5):
            ax.axhline(boundary, color='#e1e6e8', linewidth=.75)
        for i, family in enumerate(FAMILIES):
            for nfit, (color, marker, offset) in STYLES.items():
                point = points.loc[(family, split, nfit)]
                value = point['mean']
                y = 2 - i + offset
                ax.errorbar(value, y, xerr=[[value - point.lo95], [point.hi95 - value]], fmt=marker,
                            color=color, markersize=5.3, linewidth=1.35, capsize=3.0, zorder=3)
                label = f'{value:+.3f}'.replace('-', '−')
                ax.annotate(label, (point.hi95, y), xytext=(5, 0), textcoords='offset points', va='center',
                            fontsize=8.4, color=color, bbox=dict(boxstyle='square,pad=.1', fc='white', ec='none'))
        ax.set(xlim=limits, ylim=(-.48, 2.48), yticks=(2, 1, 0), yticklabels=NAMES)
        ax.xaxis.set_major_locator(MultipleLocator(.2))
        ax.tick_params(axis='both', length=0, labelsize=9, colors=secondary, pad=6)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#bbc7ca')
        ax.set_title(title, fontsize=11.5, loc='left', color=ink, fontweight='semibold', pad=10)

    fig.text(.032, .936, 'One fitted temperature versus centered log-ratio stacking', fontsize=16.5, color=ink, weight='bold')
    fig.text(.032, .88, 'Paired differences in class-summed conditional-excess Brier across 16 independent data banks',
             fontsize=10.5, color=secondary)
    handles = [Line2D([], [], marker=marker, color=color, lw=0, markersize=6, label=f'{n:,} fit examples')
               for n, (color, marker, _) in STYLES.items()]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.025, .842), ncol=2, frameon=False,
               handletextpad=.2, columnspacing=1.35, fontsize=9.)
    fig.text(.585, .118, 'Temperature pool − CLR stacking   (lower favors the temperature pool)', ha='center', fontsize=10., color=ink)
    fig.text(.032, .046, 'Bars show pointwise 95% intervals. Correlation rises in Committee and falls in Saturated. '
             'The heterogeneous shift moves the two strongest experts.', fontsize=9., color=secondary)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / 'results' / 'paired.csv')
    parser.add_argument('--out', type=Path, default=HERE / 'temperature-shift.png')
    args = parser.parse_args()
    render(args.source, args.out)
