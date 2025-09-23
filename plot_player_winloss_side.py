import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter

# Interactive display toggle
SHOW_PLOTS = True

# Seaborn theme
sns.set_theme(style="whitegrid", context="talk", palette="deep")

# Figure saving setup
FIG_ROOT = Path("figures")
(FIG_ROOT / "E4").mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

def save_fig(fig, filename: str, subdir: str = None, dpi: int = 200):
    out_dir = FIG_ROOT if not subdir else FIG_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path.as_posix(), dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {out_path}")
    return out_path.as_posix()

with open("player_winloss_side.json", "r") as f:
    json_data = json.load(f)

SUMMONER_NAME = json_data.get("summoner_name", "?")
TAGLINE = json_data.get("tagline", "?")
results = json_data.get("results", [])

# Build aggregation
if results:
    df = pd.DataFrame(results)
    # Normalize side labels
    df['side'] = df['side'].str.capitalize()

    # --- Count plot: Games by side and result (seaborn countplot) ---
    plt.figure(figsize=(7,5))
    ax = sns.countplot(
        data=df,
        x='side', hue='win', dodge=True,
        palette={True: '#2b8cbe', False: '#e34a33'},
        order=sorted(df['side'].unique())
    )
    plt.title(f"Games by Side and Result for {SUMMONER_NAME}#{TAGLINE}", fontsize=16, weight='bold')
    plt.xlabel("Side"); plt.ylabel("Games")
    # Annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%d', padding=3)
    ax.legend(title="Win", frameon=True)
    plt.tight_layout()
    save_fig(plt.gcf(), f"E4_script_{RUN_STAMP}_{SUMMONER_NAME}_{TAGLINE}_CountBySideWin.png", subdir="E4")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # --- Winrate bar with 95% CI (seaborn barplot over 0/1 win) ---
    plt.figure(figsize=(6,5))
    df['win_num'] = df['win'].astype(int)
    ax = sns.barplot(
        data=df,
        x='side', y='win_num', estimator=np.mean,
        errorbar=('ci', 95),
        hue='side', palette={'Blue': '#4a90e2', 'Red': '#e94b3c'},
        dodge=False, edgecolor='black', linewidth=0.6, legend=False
    )
    ax.set_title("Winrate by Side (%)", fontsize=16, weight='bold')
    ax.set_xlabel("Side")
    ax.set_ylabel("Winrate (%)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    # Annotate bars with percentages
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h*100:.1f}%", (p.get_x()+p.get_width()/2., h), ha='center', va='bottom', fontsize=10, xytext=(0,4), textcoords='offset points')
    plt.tight_layout()
    save_fig(plt.gcf(), f"E4_script_{RUN_STAMP}_{SUMMONER_NAME}_{TAGLINE}_WinrateBar_CI.png", subdir="E4")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # --- Winrate point plot with 95% CI (minimal) ---
    plt.figure(figsize=(6,5))
    ax = sns.pointplot(data=df, x='side', y='win_num', errorbar=('ci',95), join=False, color='#333', markers='o')
    ax.set_title("Winrate by Side (Point with 95% CI)", fontsize=16, weight='bold')
    ax.set_xlabel("Side")
    ax.set_ylabel("Winrate (%)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    # Annotate means
    means = df.groupby('side')['win_num'].mean()
    for i, side in enumerate(sorted(df['side'].unique())):
        m = means.loc[side]
        ax.text(i, m + 0.03, f"{m*100:.1f}%", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    save_fig(plt.gcf(), f"E4_script_{RUN_STAMP}_{SUMMONER_NAME}_{TAGLINE}_WinratePoint_CI.png", subdir="E4")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

else:
    print("No results in player_winloss_side.json")
