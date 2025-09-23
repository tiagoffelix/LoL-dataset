import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone

# Toggle to show plots interactively
SHOW_PLOTS = True  # Set False to disable interactive display

# Figure saving setup
FIG_ROOT = Path("figures")
(FIG_ROOT / "E1").mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

# Global seaborn theme for professional look
sns.set_theme(style="whitegrid", context="talk", palette="deep")

# Helper to annotate bars
def annotate_bars(ax, fmt="{:.2f}"):
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h):
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width()/2., h),
                        ha='center', va='bottom', fontsize=9,
                        xytext=(0,4), textcoords='offset points', color='#222')

def safe_remove_legend(ax):
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.remove()
        except Exception:
            pass

def save_fig(fig, filename: str, subdir: str = None, dpi: int = 200):
    out_dir = FIG_ROOT if not subdir else FIG_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path.as_posix(), dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {out_path}")
    return out_path.as_posix()

# Load match info (replace with your actual file if needed)
with open("arutnevjr_ajr_matches_info.json", "r") as f:
    matches_info = json.load(f)

# Use the first (most recent) match
match = matches_info[0]

# Extract item0-item6 for each player
item_cols = [f"item{i}" for i in range(7)]
item_data = {col: [match["items"][i][j] if j < len(match["items"][i]) else 0 for i in range(len(match["items"]))] for j, col in enumerate(item_cols)}

# Build DataFrame
df = pd.DataFrame({
    "champion": match["picks"],
    "position": match["position"],
    "team": match["teams"],
    "kills": match["kills"],
    "deaths": match["deaths"],
    "assists": match["assists"],
    "gold": match["gold"],
    "win": match["win"],
    **item_data
})

# Compute KDA
df["KDA"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)

# Sort champions by KDA for clearer story
df_kda = df.sort_values("KDA", ascending=False)

# --- KDA bar (sorted) ---
plt.figure(figsize=(10,5))
ax = sns.barplot(data=df_kda, x="champion", y="KDA", hue="team", dodge=True, edgecolor='black', linewidth=0.6)
ax.set_title("KDA by Champion (Single Match)", fontsize=16, weight='bold')
ax.set_xlabel("Champion")
ax.set_ylabel("KDA")
annotate_bars(ax)
ax.legend(title="Team", frameon=True)
plt.tight_layout()
save_fig(plt.gcf(), f"E1_script_{RUN_STAMP}_KDA.png", subdir="E1")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# --- Average gold per team (horizontal for readability) ---
avg_gold = df.groupby("team", as_index=False)["gold"].mean().rename(columns={"gold":"avg_gold"})
plt.figure(figsize=(6,4))
ax = sns.barplot(data=avg_gold, x="avg_gold", y="team", hue="team", dodge=False, palette="deep", edgecolor='black', linewidth=0.6)
ax.set_title("Average Gold per Team", fontsize=15, weight='bold')
ax.set_xlabel("Average Gold")
ax.set_ylabel("")
for p in ax.patches:
    w = p.get_width()
    ax.annotate(f"{w:.0f}", (w, p.get_y()+p.get_height()/2), ha='left', va='center', fontsize=9, xytext=(4,0), textcoords='offset points')
safe_remove_legend(ax)
plt.tight_layout()
save_fig(plt.gcf(), f"E1_script_{RUN_STAMP}_AvgGold.png", subdir="E1")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# --- Average damage taken per team (if available) ---
if "damageTaken" in match:
    df["damageTaken"] = match["damageTaken"]
    avg_damage_taken = df.groupby("team", as_index=False)["damageTaken"].mean().rename(columns={"damageTaken":"avg_damage_taken"})
    plt.figure(figsize=(6,4))
    ax = sns.barplot(data=avg_damage_taken, x="avg_damage_taken", y="team", hue="team", dodge=False, palette="deep", edgecolor='black', linewidth=0.6)
    ax.set_title("Average Damage Taken per Team", fontsize=15, weight='bold')
    ax.set_xlabel("Average Damage Taken")
    ax.set_ylabel("")
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f"{w:.0f}", (w, p.get_y()+p.get_height()/2), ha='left', va='center', fontsize=9, xytext=(4,0), textcoords='offset points')
    safe_remove_legend(ax)
    plt.tight_layout()
    save_fig(plt.gcf(), f"E1_script_{RUN_STAMP}_AvgDamageTaken.png", subdir="E1")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

# --- Most used items ---
items_flat = []
for col in item_cols:
    items_flat.extend(df[col].tolist())
items_flat = [i for i in items_flat if i != 0]
if items_flat:
    item_counts = (pd.Series(items_flat)
                     .value_counts()
                     .reset_index()
                     .rename(columns={"index":"itemId", 0:"count"}))
    top_items = item_counts.head(7)
    plt.figure(figsize=(8,4))
    ax = sns.barplot(data=top_items, x="itemId", y="count", palette="crest", edgecolor='black', linewidth=0.6)
    ax.set_title("Most Used Items (Top 7)", fontsize=15, weight='bold')
    ax.set_xlabel("Item ID")
    ax.set_ylabel("Count")
    annotate_bars(ax, fmt="{:.0f}")
    plt.tight_layout()
    save_fig(plt.gcf(), f"E1_script_{RUN_STAMP}_MostUsedItems.png", subdir="E1")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()