#!/usr/bin/env python3
"""Reproduce OFAT figures from raw_sensitivity_results.csv beside this script.

Run: python3 sensitivity_plots.py
Dependencies: numpy, pandas, matplotlib; scipy optional for t quantiles.
Outputs: figures/*.png (400 dpi), *.pdf, plot_statistics.csv, validation.json.
Bands are pointwise two-sided Student-t confidence intervals for each mean,
not prediction intervals or confidence intervals for paired differences.
No smoothing or filtering is performed. Original files are read-only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

ALGORITHMS = ['Greedy', 'Rollout-Time', 'Rollout-Benefit']
STYLES = [('#9ABBEF', 's'), ('#9DD49D', 'o'), ('#BAABCE', '^')]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor', choices=['lambda', 'alpha', 'P01'], help='Regenerate only this factor')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    source = base / 'raw_sensitivity_results.csv'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source, encoding='utf-8-sig')
    keys = ['factor', 'level', 'algorithm']
    require(len(df) == 4410, 'Expected 4410 raw runs')
    require(set(df.factor) == {'lambda', 'alpha', 'P01'}, 'Unexpected factors')
    require(set(df.algorithm) == set(ALGORITHMS), 'Unexpected algorithms')
    require(np.isfinite(df.objective).all(), 'Nonfinite objective')
    require(not df.duplicated(keys + ['scenario_id']).any(), 'Duplicate scenario')
    require((df.termination == 'completed').all() and (df.unfinished_tasks == 0).all(), 'Incomplete runs')
    pairing = set(zip(df.scenario_id, df.environment_seed))
    require(len(pairing) == 30, 'Scenario/seed pairing differs')
    for key, group in df.groupby(keys):
        require(len(group) == 30 and set(zip(group.scenario_id, group.environment_seed)) == pairing,
                f'Missing or unpaired observations: {key}')
    counts = df.groupby('factor')['level'].nunique().to_dict()
    require(counts == {'P01': 7, 'alpha': 21, 'lambda': 21}, 'Unexpected level counts')
    require(np.allclose(sorted(df[df.factor == 'lambda'].level.unique()), np.linspace(.9, 1.1, 21)), 'Unexpected fatigue levels')
    road = df[df.factor == 'P01']
    require(np.allclose(road.level, road.P_block), 'P01 level must mean applied probability')
    try:
        from scipy.stats import t
        critical = float(t.ppf(.975, 29))
    except ImportError:
        critical = 2.045229642132703  # Student-t 97.5th percentile, df=29; not experimental data.
    summary = df.groupby(keys).objective.agg(n='count', mean='mean', sd='std').reset_index()
    summary['se'] = summary.sd / np.sqrt(summary.n)
    summary['ci95_half_width'] = critical * summary.se
    summary['ci95_lower'] = summary['mean'] - summary.ci95_half_width
    summary['ci95_upper'] = summary['mean'] + summary.ci95_half_width
    # Cross-check every group against independent standard-library calculations.
    for row in summary.itertuples():
        values = df[(df.factor == row.factor) & (df.level == row.level) & (df.algorithm == row.algorithm)].objective.tolist()
        require(math.isclose(row.mean, statistics.mean(values), abs_tol=1e-12), 'Mean mismatch')
        require(math.isclose(row.sd, statistics.stdev(values), abs_tol=1e-12), 'SD mismatch')
    existing = pd.read_csv(base / 'sensitivity_summary.csv', encoding='utf-8-sig')
    checked = summary.merge(existing, on=keys, suffixes=('', '_existing'), validate='one_to_one')
    require(len(checked) == len(summary) == len(existing), 'Summary groups differ')
    deviations = {}
    for col in ['n', 'mean', 'sd', 'se', 'ci95_lower', 'ci95_upper', 'ci95_half_width']:
        deviations[col] = float(np.max(np.abs(checked[col] - checked[col + '_existing'])))
        require(np.allclose(checked[col], checked[col + '_existing'], rtol=1e-10, atol=1e-12), f'Existing summary differs: {col}')
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'] if f in available), 'DejaVu Serif')
    plt.rcParams.update({'font.family': font, 'font.size': 15, 'axes.labelsize': 18,
                         'axes.linewidth': 1.8, 'axes.grid': False, 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white'})
    out = base / 'figures'
    out.mkdir(exist_ok=True)
    configs = [('lambda', 'fatigue', 'Fatigue Rate Adjustment Percentage (%)'),
               ('alpha', 'alpha', r'Mutual-aid Parameter $\alpha$'),
               ('P01', 'P01', r'Road Blocked-transition Probability $P_{01}$')]
    for factor, name, label in configs:
        if args.factor and factor != args.factor:
            continue
        fig, ax = plt.subplots(figsize=(10, 7.5))
        fig.subplots_adjust(left=.13, right=.97, bottom=.15, top=.80)
        for algorithm, (color, marker) in zip(ALGORITHMS, STYLES):
            data = summary[(summary.factor == factor) & (summary.algorithm == algorithm)].sort_values('level')
            x = data.level.to_numpy()
            if factor == 'lambda':
                x = (x - 1) * 100
            y = data['mean'].to_numpy()
            ax.fill_between(x, data.ci95_lower.to_numpy(), data.ci95_upper.to_numpy(), color=color, alpha=.20, linewidth=0)
            line, = ax.plot(x, y, color=color, marker=marker, markersize=6.5, linewidth=1.05,
                            markeredgewidth=.5, label=algorithm)
            require(np.array_equal(line.get_ydata(), y), 'Plotted data differs')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(which='major', direction='out', length=7, width=1.6, pad=7)
        ax.tick_params(which='minor', direction='out', length=3.5, width=1.2)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel(label, labelpad=15)
        ax.set_ylabel('Weighted Total Rescue Time', labelpad=14)
        ax.margins(x=.07, y=.09)
        if factor == 'lambda':
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+.0f}' if v else '0'))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        elif factor == 'alpha':
            ax.set_xticks(np.linspace(.18, .22, 5))
        else:
            ax.set_xticks(np.linspace(.2, .6, 5))
        legend = ax.legend(loc='lower right', bbox_to_anchor=(1, 1.025), frameon=True,
                           fancybox=False, edgecolor='.35', fontsize=14, borderpad=.5)
        legend.get_frame().set_linewidth(.7)
        for extension in ['png', 'pdf']:
            fig.savefig(out / f'{name}_sensitivity_95CI.{extension}', dpi=400, facecolor='white')
        if factor == 'alpha':
            fig.savefig(out / 'Mutual Aid Capacity Sensitivity Analysis.jpg', dpi=400,
                        facecolor='white', pil_kwargs={'quality': 95})
        plt.close(fig)
    summary.to_csv(out / 'plot_statistics.csv', index=False)
    require(hashlib.sha256(source.read_bytes()).hexdigest() == original_hash, 'Raw data changed')
    report = {'rows': len(df), 'level_counts': counts, 'paired_scenarios_per_group': 30,
              'groups': len(summary), 'font': font, 't_critical_df29': critical,
              'source_sha256': original_hash, 'max_absolute_summary_differences': deviations,
              'CI': 'pointwise mean +/- t(0.975,29) * sample SD / sqrt(30)',
              'P01_axis': 'level (applied probability, equal to P_block)',
              'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}}
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Figures saved to {out}')

if __name__ == '__main__':
    main()
