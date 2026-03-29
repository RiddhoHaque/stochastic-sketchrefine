#!/usr/bin/env python3
"""
Generate benchmark plots from Evaluation/benchmark_results.json.

Run from the project root or from DemoScripts:
    python DemoScripts/plot_benchmark_results.py

Output:
    Evaluation/price_normalized_rmse_percentile_lines_<year>.png  -- per-year percentile price-normalized RMSE charts
    Evaluation/runtime_comparison.png            -- scenario generation runtimes by year
"""
import os
import sys
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INPUT_PATH = os.path.join(_PROJECT_ROOT, 'Evaluation', 'benchmark_results.json')
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'Evaluation')

FONT_TITLE  = 16
FONT_AXIS   = 14
FONT_TICK   = 13
FONT_LEGEND = 13

PALETTE = {'GBM': '#4C72B0', 'GARCH-FHS': '#DD8452'}


def _grid_shape(count):
    if count <= 1:
        return 1, 1
    if count == 2:
        return 1, 2
    if count <= 4:
        return 2, 2
    return int(np.ceil(count / 2.0)), 2


def _build_series(method_price_normalized_rmse, year_str, percentile_str,
                  target_sas):
    x_vals, y_vals = [], []
    for sa in target_sas:
        val = (
            method_price_normalized_rmse
            .get(year_str, {})
            .get(str(sa), {})
            .get(percentile_str)
        )
        if val is None:
            continue
        x_vals.append(sa)
        y_vals.append(val)
    return x_vals, y_vals


def _plot_year_percentile_lines(year, target_sas, percentiles,
                                gbm_price_normalized_rmse,
                                garch_price_normalized_rmse):
    year_str = str(year)
    n_rows, n_cols = _grid_shape(len(percentiles))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.2 * n_cols, 4.4 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)
    flat_axes = list(axes_arr.flatten())

    all_vals = []
    for percentile in percentiles:
        percentile_str = str(percentile)
        for method_rmse in (
            gbm_price_normalized_rmse, garch_price_normalized_rmse
        ):
            _, y_vals = _build_series(method_rmse, year_str, percentile_str, target_sas)
            all_vals.extend(y_vals)

    ylim = (0.0, 0.3)
    positive_vals = [v for v in all_vals if v > 0]
    if not positive_vals:
        ylim = None

    for idx, percentile in enumerate(percentiles):
        ax = flat_axes[idx]
        percentile_str = str(percentile)
        gbm_x, gbm_y = _build_series(
            gbm_price_normalized_rmse, year_str, percentile_str, target_sas
        )
        garch_x, garch_y = _build_series(
            garch_price_normalized_rmse, year_str, percentile_str, target_sas
        )

        if gbm_y:
            ax.plot(
                gbm_x, gbm_y,
                color=PALETTE['GBM'],
                marker='o',
                linewidth=2.2,
                markersize=5.5,
                label='GBM',
            )
        if garch_y:
            ax.plot(
                garch_x, garch_y,
                color=PALETTE['GARCH-FHS'],
                marker='s',
                linewidth=2.2,
                markersize=5.5,
                label='GARCH-FHS',
            )
        if not gbm_y and not garch_y:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONT_TICK)

        ax.set_title(f'P{percentile}', fontsize=FONT_TITLE)
        ax.set_xticks(target_sas)
        ax.tick_params(axis='x', labelsize=FONT_TICK)
        ax.tick_params(axis='y', labelsize=FONT_TICK)
        ax.grid(True, which='major', axis='both', alpha=0.25, linewidth=0.8)
        if ylim is not None:
            ax.set_ylim(ylim)

    for ax in flat_axes[len(percentiles):]:
        ax.axis('off')

    for row_axes in axes_arr:
        row_axes[0].set_ylabel('Price-normalized RMSE', fontsize=FONT_AXIS)
    for ax in axes_arr[-1]:
        if ax.axison:
            ax.set_xlabel('Sell-after days', fontsize=FONT_AXIS)

    legend_handles = [
        Line2D([0], [0], color=PALETTE['GBM'], marker='o', linewidth=2.2, label='GBM'),
        Line2D([0], [0], color=PALETTE['GARCH-FHS'], marker='s', linewidth=2.2, label='GARCH-FHS'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        fontsize=FONT_LEGEND,
    )
    fig.suptitle(
        f'GBM vs GARCH-FHS Price-Normalized RMSE by Sell-After Day ({year}-{year+1})',
        fontsize=FONT_TITLE + 2, fontweight='bold', y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    output_path = os.path.join(
        OUTPUT_DIR, f'price_normalized_rmse_percentile_lines_{year}.png'
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Year-specific price-normalized RMSE line charts saved to {output_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH) as f:
        results = json.load(f)

    years = results['years']
    no_of_scenarios = results['no_of_scenarios']
    target_sas = results['target_sell_afters']
    percentiles = [
        p for p in results['price_normalized_rmse_percentiles']
        if int(p) < 100
    ]
    gbm_price_normalized_rmse = results['gbm_price_normalized_rmse']
    garch_price_normalized_rmse = results['garch_price_normalized_rmse']
    gbm_runtimes = {int(y): t for y, t in results['gbm_runtimes'].items()}
    garch_runtimes = {int(y): t for y, t in results['garch_runtimes'].items()}

    for year in years:
        _plot_year_percentile_lines(
            year, target_sas, percentiles,
            gbm_price_normalized_rmse, garch_price_normalized_rmse,
        )

    common_years = sorted(set(gbm_runtimes) & set(garch_runtimes))
    if common_years:
        gbm_t = [gbm_runtimes[y] for y in common_years]
        garch_t = [garch_runtimes[y] for y in common_years]

        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        x, w = np.arange(len(common_years)), 0.35
        ax2.bar(x - w / 2, gbm_t,   w, label='GBM',       color=PALETTE['GBM'], alpha=0.75)
        ax2.bar(x + w / 2, garch_t, w, label='GARCH-FHS', color=PALETTE['GARCH-FHS'], alpha=0.75)
        ax2.axhline(np.mean(gbm_t), color=PALETTE['GBM'], linestyle='--', linewidth=1.2,
                    label=f'GBM mean {np.mean(gbm_t):.1f}s')
        ax2.axhline(np.mean(garch_t), color=PALETTE['GARCH-FHS'], linestyle='--', linewidth=1.2,
                    label=f'GARCH-FHS mean {np.mean(garch_t):.1f}s')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{y}-{y+1}' for y in common_years], fontsize=FONT_TICK)
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
        print('No common years for runtime comparison; skipping chart.')

    method_w = 12
    pct_cols = [str(p) for p in percentiles]
    sa_w, yr_w = 18, 12
    divider = '-' * (sa_w + yr_w + len(pct_cols) * method_w * 2)

    lines = []
    lines.append('\n' + '=' * len(divider))
    lines.append('Price-Normalized RMSE Summary')
    lines.append('=' * len(divider))

    header1 = f'{"Sell After (Days)":>{sa_w}}{"Year":>{yr_w}}'
    for pct in pct_cols:
        header1 += f'{("P" + pct):^{method_w * 2}}'
    header2 = f'{"":>{sa_w}}{"":>{yr_w}}'
    for _ in pct_cols:
        header2 += f'{"GBM":>{method_w}}{"GARCH":>{method_w}}'
    lines.append(header1)
    lines.append(header2)
    lines.append(divider)

    for sa in target_sas:
        sa_str = str(sa)
        first_row = True
        for year in years:
            year_str = str(year)
            row = f'{(str(sa) if first_row else ""):>{sa_w}}{f"{year}-{year+1}":>{yr_w}}'
            first_row = False
            gbm_sa = gbm_price_normalized_rmse.get(year_str, {}).get(sa_str, {})
            garch_sa = garch_price_normalized_rmse.get(year_str, {}).get(sa_str, {})
            for pct in pct_cols:
                gbm_val = gbm_sa.get(pct)
                garch_val = garch_sa.get(pct)
                gbm_str = f'{gbm_val:.4f}' if gbm_val is not None else 'N/A'
                garch_str = f'{garch_val:.4f}' if garch_val is not None else 'N/A'
                row += f'{gbm_str:>{method_w}}{garch_str:>{method_w}}'
            lines.append(row)
        lines.append(divider)

    table_text = '\n'.join(lines)
    print(table_text)

    table_path = os.path.join(OUTPUT_DIR, 'summary_statistics.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table_text + '\n')
    print(f'Price-normalized RMSE summary saved to {table_path}')


if __name__ == '__main__':
    main()
