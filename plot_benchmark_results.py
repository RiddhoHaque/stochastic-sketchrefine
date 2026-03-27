#!/usr/bin/env python3
"""
Generate benchmark plots from Evaluation/benchmark_results.json.

Run from project root:
    python plot_benchmark_results.py

Output:
    Evaluation/rmse_boxplots.png       -- 3 x 4 grid: sell_after x year-pair
    Evaluation/runtime_comparison.png  -- scenario generation runtimes by year
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_PATH = os.path.join('Evaluation', 'benchmark_results.json')
OUTPUT_DIR = 'Evaluation'

FONT_TITLE  = 16
FONT_AXIS   = 14
FONT_TICK   = 13
FONT_LEGEND = 13

PALETTE = {'GBM': '#4C72B0', 'GARCH-FHS': '#DD8452'}

BOX_WIDTH = 0.55   # wider boxes (default is 0.5)


def _make_boxplot(ax, gbm_vals, garch_vals, title, show_ylabel: bool,
                  ylabel_text: str, ylim):
    data, tick_labels, colors = [], [], []
    for vals, label in [(gbm_vals, 'GBM'), (garch_vals, 'GARCH-FHS')]:
        if vals:
            data.append(vals)
            tick_labels.append(label)
            colors.append(PALETTE[label])
    if not data:
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONT_TICK)
        return
    bp = ax.boxplot(data, labels=tick_labels, patch_artist=True,
                    widths=BOX_WIDTH, showfliers=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    if show_ylabel:
        ax.set_ylabel(ylabel_text, fontsize=FONT_AXIS)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH) as f:
        results = json.load(f)

    years             = results['years']              # e.g. [2015, 2017, 2019, 2021]
    no_of_scenarios   = results['no_of_scenarios']
    target_sas        = results['target_sell_afters'] # e.g. [10, 100, 200]
    gbm_errors        = results['gbm_errors']         # {str(year): {str(sa): {ticker: err}}}
    garch_errors      = results['garch_errors']
    gbm_runtimes      = {int(y): t for y, t in results['gbm_runtimes'].items()}
    garch_runtimes    = {int(y): t for y, t in results['garch_runtimes'].items()}

    n_rows = len(target_sas)   # 3
    n_cols = len(years)        # 4

    # Compute global y-axis limits across all subplots (log scale)
    all_vals = []
    for sa in target_sas:
        sa_str = str(sa)
        for year in years:
            year_str = str(year)
            all_vals.extend(gbm_errors.get(year_str, {}).get(sa_str, {}).values())
            all_vals.extend(garch_errors.get(year_str, {}).get(sa_str, {}).values())

    positive_vals = [v for v in all_vals if v > 0]
    if positive_vals:
        ylim = (10 ** (np.floor(np.log10(min(positive_vals))) - 0.1),
                10 ** (np.ceil(np.log10(max(positive_vals)))  + 0.1))
    else:
        ylim = None

    # ---- 3 × 4 error boxplot grid ----
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        sharey=False,
    )
    axes_2d = np.array(axes).reshape(n_rows, n_cols)

    for row, sa in enumerate(target_sas):
        sa_str      = str(sa)
        ylabel_text = f'Rate-of-return error\nSA = {sa}d'
        for col, year in enumerate(years):
            ax         = axes_2d[row, col]
            year_str   = str(year)
            show_ylabel = (col == 0)

            gbm_vals   = list(
                gbm_errors.get(year_str, {}).get(sa_str, {}).values()
            )
            garch_vals = list(
                garch_errors.get(year_str, {}).get(sa_str, {}).values()
            )

            title = f'{year}–{year+1}'
            _make_boxplot(ax, gbm_vals, garch_vals, title,
                          show_ylabel, ylabel_text, ylim)

    fig.suptitle(
        f'GBM vs GARCH-FHS  |  Rate-of-Return Prediction Error  '
        f'({no_of_scenarios} scenarios)',
        fontsize=FONT_TITLE + 2, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    rmse_path = os.path.join(OUTPUT_DIR, 'rmse_boxplots.png')
    fig.savefig(rmse_path, dpi=150)
    plt.close(fig)
    print(f'Error boxplots saved to {rmse_path}')

    # ---- runtime bar chart ----
    common_years = sorted(set(gbm_runtimes) & set(garch_runtimes))
    if common_years:
        gbm_t   = [gbm_runtimes[y]   for y in common_years]
        garch_t = [garch_runtimes[y] for y in common_years]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        x, w = np.arange(len(common_years)), 0.35
        ax2.bar(x - w / 2, gbm_t,   w, label='GBM',       color='#4C72B0', alpha=0.75)
        ax2.bar(x + w / 2, garch_t, w, label='GARCH-FHS', color='#DD8452', alpha=0.75)
        ax2.axhline(np.mean(gbm_t),   color='#4C72B0', linestyle='--', linewidth=1.2,
                    label=f'GBM mean {np.mean(gbm_t):.1f}s')
        ax2.axhline(np.mean(garch_t), color='#DD8452', linestyle='--', linewidth=1.2,
                    label=f'GARCH-FHS mean {np.mean(garch_t):.1f}s')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{y}–{y+1}' for y in common_years],
                            fontsize=FONT_TICK)
        ax2.set_xlabel('Year pair', fontsize=FONT_AXIS)
        ax2.set_ylabel('Runtime (s)', fontsize=FONT_AXIS)
        ax2.tick_params(axis='y', labelsize=FONT_TICK)
        ax2.set_title(f'Scenario Generation Runtime ({no_of_scenarios} scenarios)',
                      fontsize=FONT_TITLE)
        ax2.legend(fontsize=FONT_LEGEND)
        plt.tight_layout()
        rt_path = os.path.join(OUTPUT_DIR, 'runtime_comparison.png')
        fig2.savefig(rt_path, dpi=150)
        plt.close(fig2)
        print(f'Runtime chart saved to {rt_path}')
    else:
        print('No common years for runtime comparison — skipping chart.')

    # ---- summary statistics table ----
    # Layout: SA | Year | stat(GBM) stat(GARCH) | stat(GBM) stat(GARCH) | ...
    stat_cols = ['P25', 'Median', 'P75', 'P90', 'P95']
    mth_w  = 11   # width per method value within a stat group
    grp_w  = 2 * mth_w   # width of a stat group (GBM + GARCH side by side)
    sa_w, yr_w = 20, 12
    divider = '-' * (sa_w + yr_w + len(stat_cols) * grp_w)

    def _stats(vals):
        if not vals:
            return ('N/A',) * len(stat_cols)
        a = np.array(vals)
        return (f'{np.percentile(a, 25):.2f}',
                f'{np.median(a):.2f}',
                f'{np.percentile(a, 75):.2f}',
                f'{np.percentile(a, 90):.2f}',
                f'{np.percentile(a, 95):.2f}')

    lines = []
    lines.append('\n' + '=' * len(divider))
    lines.append('Rate-of-Return Prediction Error — Summary Statistics')
    lines.append('=' * len(divider))

    # Header row 1: stat group labels
    stat_header1 = f'{"Sell After (Days)":>{sa_w}}{"Year":>{yr_w}}'
    for sc in stat_cols:
        stat_header1 += f'{sc:^{grp_w}}'
    # Header row 2: GBM / GARCH under each stat group
    stat_header2 = f'{"":>{sa_w}}{"":>{yr_w}}'
    for _ in stat_cols:
        stat_header2 += f'{"GBM":>{mth_w}}{"GARCH":>{mth_w}}'
    lines.append(stat_header1)
    lines.append(stat_header2)
    lines.append(divider)

    for sa in target_sas:
        sa_str = str(sa)
        first_row = True
        for year in years:
            year_str = str(year)
            label = f'{year}–{year+1}'

            gv = list(gbm_errors.get(year_str, {}).get(sa_str, {}).values())
            cv = list(garch_errors.get(year_str, {}).get(sa_str, {}).values())
            gs = _stats(gv)
            cs = _stats(cv)

            sa_label = str(sa) if first_row else ''
            first_row = False
            row = f'{sa_label:>{sa_w}}{label:>{yr_w}}'
            for g_val, c_val in zip(gs, cs):
                row += f'{g_val:>{mth_w}}{c_val:>{mth_w}}'
            lines.append(row)
        lines.append(divider)

    table_text = '\n'.join(lines)
    print(table_text)

    table_path = os.path.join(OUTPUT_DIR, 'summary_statistics.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table_text + '\n')
    print(f'Summary statistics saved to {table_path}')


if __name__ == '__main__':
    main()
