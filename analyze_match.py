import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load match info (replace with your actual file if needed)
with open("arutnevjr_ajr_matches_info.json", "r") as f:
    matches_info = json.load(f)

# Get the last match (first in the list)
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


# KDA calculation
df["KDA"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)


# Average gold per team
avg_gold = df.groupby("team")["gold"].mean()


# Average damage taken per team (if available)
if "damageTaken" in match:
    df["damageTaken"] = match["damageTaken"]
    avg_damage_taken = df.groupby("team")["damageTaken"].mean()
else:
    avg_damage_taken = "damageTaken not available in data"


# Most used items (from item0-item6 columns)
items = []
for col in item_cols:
    items.extend(df[col].tolist())
items = [item for item in items if item != 0]
most_used_items = pd.Series(items).value_counts().head(5)



# KDA bar plot with value labels
plt.figure(figsize=(8, 4))
ax = sns.barplot(x="champion", y="KDA", hue="team", data=df)
plt.title("KDA dos jogadores")
plt.ylabel("KDA")
plt.xlabel("Champion")
plt.legend()
for p in ax.patches:
    height = p.get_height()
    if not pd.isna(height):
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()


# Average gold per team bar plot with value labels
plt.figure(figsize=(6, 4))
ax = avg_gold.plot(kind="bar", color=["blue", "red"])
plt.title("Average Gold per Team")
plt.ylabel("Average Gold")
plt.xlabel("Team")
for i, v in enumerate(avg_gold):
    ax.annotate(f'{v:.0f}', (i, v), ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()


# Average damage taken per team bar plot with value labels (if available)
if isinstance(avg_damage_taken, pd.Series):
    plt.figure(figsize=(6, 4))
    ax = avg_damage_taken.plot(kind="bar", color=["blue", "red"])
    plt.title("Average Damage Taken per Team")
    plt.ylabel("Average Damage Taken")
    plt.xlabel("Team")
    for i, v in enumerate(avg_damage_taken):
        ax.annotate(f'{v:.0f}', (i, v), ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.show()


# Most used items bar plot with value labels
plt.figure(figsize=(8, 4))
ax = most_used_items.plot(kind="bar")
plt.title("Most Used Items in the Game (item0-item6)")
plt.ylabel("Count")
plt.xlabel("Item ID")
for i, v in enumerate(most_used_items):
    ax.annotate(f'{v}', (i, v), ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()