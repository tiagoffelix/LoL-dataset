import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Interactive display toggle
SHOW_PLOTS = True

# Seaborn theme
sns.set_theme(style="whitegrid", context="talk", palette="deep")

# Figure saving setup
FIG_ROOT = Path("figures")
(FIG_ROOT / "E4").mkdir(parents=True, exist_ok=True)
RUN_DATE = datetime.now(timezone.utc).date().isoformat()

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
    # Win column already boolean

    # Summary counts
    summary = (df
               .groupby(['side', 'win'])
               .size()
               .reset_index(name='count'))

    # Pivot for stacked bar
    pivot = (summary
             .pivot(index='side', columns='win', values='count')
             .fillna(0)
             .rename(columns={True:'Wins', False:'Losses'}))

    pivot['Total'] = pivot['Wins'] + pivot['Losses']
    pivot['Winrate'] = (pivot['Wins'] / pivot['Total'] * 100).round(1)

    # --- Stacked bar (more readable than pie for side comparison) ---
    plt.figure(figsize=(7,5))
    bottom = None
    colors = {'Wins':'#2b8cbe', 'Losses':'#e34a33'}
    for col in ['Wins', 'Losses']:
        plt.bar(pivot.index, pivot[col], bottom=bottom, label=col, color=colors[col], edgecolor='black', linewidth=0.6)
        if bottom is None:
            bottom = pivot[col].copy()
        else:
            bottom += pivot[col]
    plt.title(f"Win/Loss by Side for {SUMMONER_NAME}#{TAGLINE}", fontsize=16, weight='bold')
    plt.ylabel("Games")
    plt.xlabel("Side")
    # Annotate totals & winrate
    for i, side in enumerate(pivot.index):
        total = pivot.loc[side, 'Total']
        wr = pivot.loc[side, 'Winrate']
        plt.text(i, total + 0.2, f"Total {total}\nWR {wr}%", ha='center', va='bottom', fontsize=10, weight='bold')
    plt.legend(title="Result", frameon=True)
    plt.tight_layout()
    # Save figure
    save_fig(plt.gcf(), f"E4_script_{RUN_DATE}_{SUMMONER_NAME}_{TAGLINE}_StackedBar.png", subdir="E4")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # --- Side share donut (overall distribution) ---
    side_totals = pivot['Total']
    plt.figure(figsize=(6,6))
    wedges, texts, autotexts = plt.pie(side_totals, labels=[f"{s} ({v})" for s,v in side_totals.items()], autopct='%1.1f%%', startangle=90, colors=['#4a90e2','#e94b3c'], wedgeprops={'edgecolor':'white'})
    centre_circle = plt.Circle((0,0),0.55,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    total_games = int(side_totals.sum())
    plt.title(f"Side Distribution (n={total_games})", fontsize=15, weight='bold')
    plt.tight_layout()
    # Save figure
    save_fig(plt.gcf(), f"E4_script_{RUN_DATE}_{SUMMONER_NAME}_{TAGLINE}_Donut.png", subdir="E4")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # --- Winrate bar with confidence style (approx, Wilson not computed) ---
    plt.figure(figsize=(6,5))
    ax = sns.barplot(x=pivot.index, y=pivot['Winrate'], palette=['#4a90e2','#e94b3c'], edgecolor='black', linewidth=0.6)
    ax.set_title("Winrate by Side (%)", fontsize=16, weight='bold')
    ax.set_xlabel("Side")
    ax.set_ylabel("Winrate (%)")
    for p, wr in zip(ax.patches, pivot['Winrate']):
        h = p.get_height()
        ax.annotate(f"{wr}%", (p.get_x()+p.get_width()/2., h), ha='center', va='bottom', fontsize=10, xytext=(0,4), textcoords='offset points')
    plt.ylim(0, 100)
    plt.tight_layout()
    # Save figure
    save_fig(plt.gcf(), f"E4_script_{RUN_DATE}_{SUMMONER_NAME}_{TAGLINE}_WinrateBar.png", subdir="E4")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

else:
    print("No results in player_winloss_side.json")
