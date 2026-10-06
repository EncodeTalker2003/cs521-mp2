#!/usr/bin/env python3
"""Plot measured data only; no placeholder performance numbers."""
import argparse
import csv
from pathlib import Path

VARIANTS = ('eager', 'inductor', 'jax', 'cuda')
LABELS = {'eager': 'PyTorch eager', 'inductor': 'Inductor', 'jax': 'JAX JIT', 'cuda': 'CUDA tiled'}
COLORS = {'eager': '#1f77b4', 'inductor': '#ff7f0e', 'jax': '#2ca02c', 'cuda': '#d62728'}


def main(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(directory / 'summary.csv') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit('No successful measurements to plot.')
    sizes = sorted({int(r['H']) for r in rows})
    filters = sorted({int(r['KH']) for r in rows})
    metrics = {
        'gpu_span_mean_us': ('GPU span per inference (us)', 'gpu_span_by_input.png'),
        'gpu_kernel_sum_us': ('Sum of GPU kernel durations per inference (us)', 'gpu_kernels_by_input.png'),
        'steady_wall_median_us': ('Synchronized wall time, median (us)', 'wall_by_input.png'),
        'host_compile_s': ('Host compile/build time (s; scopes differ)', 'compile_by_input.png'),
    }
    for metric, (ylabel, filename) in metrics.items():
        fig, axes = plt.subplots(1, len(filters), figsize=(5 * len(filters), 4), squeeze=False)
        for ax, filt in zip(axes[0], filters):
            for variant in VARIANTS:
                subset = sorted([r for r in rows if r['variant'] == variant and int(r['KH']) == filt
                                 and r[metric] != ''], key=lambda r: int(r['H']))
                if subset:
                    ax.plot([int(r['H']) for r in subset], [float(r[metric]) for r in subset],
                            marker='o', label=LABELS[variant], color=COLORS[variant])
            ax.set(title=f'Filter {filt} x {filt}', xlabel='Input H = W', ylabel=ylabel)
            ax.set_xticks(sizes)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for variant in VARIANTS:
            subset = sorted([r for r in rows if r['variant'] == variant and int(r['H']) == size],
                            key=lambda r: int(r['KH']))
            if subset:
                ax.plot([int(r['KH']) for r in subset], [float(r['gpu_span_mean_us']) for r in subset],
                        marker='o', label=LABELS[variant], color=COLORS[variant])
        ax.set(title=f'Input {size} x {size}', xlabel='Filter KH = KW', ylabel='GPU span per inference (us)')
        ax.set_xticks(filters)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'gpu_span_by_filter.png', dpi=180)
    plt.close(fig)
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Shape $(H,K_H)$ & Implementation & Compile (s) & Kernels ($\mu$s) & Span ($\mu$s) & Wall ($\mu$s) \\',
             r'\midrule']
    for row in sorted(rows, key=lambda r: (int(r['H']), int(r['KH']), VARIANTS.index(r['variant']))):
        compile_time = f"{float(row['host_compile_s']):.3f}" if row['host_compile_s'] else 'N/A'
        lines.append(f"({row['H']}, {row['KH']}) & {LABELS[row['variant']]} & {compile_time} & "
                     f"{float(row['gpu_kernel_sum_us']):.2f} & {float(row['gpu_span_mean_us']):.2f} & "
                     f"{float(row['steady_wall_median_us']):.2f} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (directory / 'report_table.tex').write_text('\n'.join(lines) + '\n')
    print(f'Saved 5 plots and report_table.tex in {directory}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    main(p.parse_args().directory)
