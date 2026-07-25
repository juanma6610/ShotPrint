"""
Federated convergence figure for Section 4.4.

Reads the four per-round federated metrics CSVs and produces a single
two-panel figure (Brier on the left, log-loss on the right) with the
centralised baseline drawn as a horizontal reference.

Output:
    figures/federated_convergence.pdf
    figures/federated_convergence.png

Run from the project root:
    python src/plot_federated_convergence.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ----- Style — keep consistent with thesis_results.py -----------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FED_DIR = Path('results/federated')
FIG_DIR = Path('figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Centralised baseline (from results/headline_metrics.csv, XGBoost full row)
CENTRAL_BRIER  = 0.219
CENTRAL_LOGLOSS = 0.6283

# Each entry: (csv_filename, label, colour, linestyle, x-axis = "round" or "n_trees")
RUNS = [
    ('federated_bagging_team_metrics.csv', 'Bagging — by-team', '#1f4e79', '-',  'n_trees'),
    ('federated_bagging_iid_metrics.csv',  'Bagging — IID',     '#5b9bd5', '-',  'n_trees'),
    ('federated_cyclic_team_metrics.csv',  'Cyclic — by-team',  '#c0504d', '--', 'n_trees'),
    ('federated_cyclic_iid_metrics.csv',   'Cyclic — IID',      '#e67c73', '--', 'n_trees'),
]


def load_runs():
    runs = []
    for fname, label, color, ls, x_key in RUNS:
        df = pd.read_csv(FED_DIR / fname)
        runs.append({'df': df, 'label': label, 'color': color, 'ls': ls, 'x_key': x_key})
    return runs


def annotate_best(ax, df, metric_col, color, label):
    """Mark the round of minimum metric value with a small dot + annotation."""
    idx = df[metric_col].idxmin()
    n_trees = df.loc[idx, 'n_trees']
    val = df.loc[idx, metric_col]
    ax.plot(n_trees, val, 'o', color=color, markersize=6, zorder=4,
            markeredgecolor='white', markeredgewidth=1.2)


def plot_panel(ax, runs, metric_col, central, ylabel, title):
    for run in runs:
        df = run['df']
        ax.plot(df['n_trees'], df[metric_col],
                color=run['color'], linestyle=run['ls'], linewidth=1.8,
                label=run['label'])
        # Mark best round
        annotate_best(ax, df, metric_col, run['color'], run['label'])

    # FIX 1: Use a uniform label for the shared legend setup
    ax.axhline(central, color='red', linestyle=':', linewidth=1.4,
               label='Centralised baseline')

    # FIX 2: Inline text annotation using a blended coordinate system
    # (x=0.02 is 2% from the left boundary; y is in absolute data values)
    ax.text(0.02, central, f'Centralised: {central:.4f}', color='red',
            transform=ax.get_yaxis_transform(), va='bottom', ha='left',
            fontsize=9, fontweight='semibold')

    ax.set_xlabel('Number of trees in global ensemble')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

def main():
    runs = load_runs()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    plot_panel(axes[0], runs, 'brier',   CENTRAL_BRIER,
               ylabel='Brier score (test)',
               title='Calibration')
    plot_panel(axes[1], runs, 'logloss', CENTRAL_LOGLOSS,
               ylabel='Logarithmic loss (test)',
               title='Log-loss')

    # Single legend across both panels — placed below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04),
               frameon=False)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    pdf_path = FIG_DIR / 'federated_convergence.pdf'
    png_path = FIG_DIR / 'federated_convergence.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f'Saved {pdf_path}')
    print(f'Saved {png_path}')

    # Print best-round summary for the section table sanity-check
    print('\nBest-round summary (verify against Table 4.x):')
    print(f'{"Configuration":<28} {"Best rd":>8} {"Trees":>7} {"Brier":>8} {"Logloss":>9} {"AUC":>8}')
    print('-' * 72)
    for run in runs:
        df = run['df']
        idx = df['brier'].idxmin()
        row = df.loc[idx]
        print(f'{run["label"]:<28} {int(row["round"]):>8d} {int(row["n_trees"]):>7d} '
              f'{row["brier"]:>8.4f} {row["logloss"]:>9.4f} {row["auc"]:>8.4f}')


if __name__ == '__main__':
    main()
