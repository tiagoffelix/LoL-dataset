import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load match info (replace with your actual file if needed)
with open("Speazyy_EUW_matches_info.json", "r") as f:
    matches_info = json.load(f)

# Get the last match (first in the list)
match = matches_info[0]

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
    "challenges": match["challenges"],
})

# KDA calculation
df["KDA"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)

# Average gold per team
avg_gold = df.groupby("team")["gold"].mean()

# Average damage taken per team (if available)
if "damageTaken" in df["challenges"][0]:
    df["damageTaken"] = [c.get("damageTaken", 0) for c in df["challenges"]]
    avg_damage_taken = df.groupby("team")["damageTaken"].mean()
else:
    avg_damage_taken = "damageTaken not available in data"

# Most used items (if available)
item_cols = [f"item{i}" for i in range(0, 7)]
items = []
for c in df["challenges"]:
    for col in item_cols:
        if col in c:
            items.append(c[col])
most_used_items = pd.Series(items).value_counts().head(5)

print("KDA dos jogadores:")
print(df[["champion", "position", "team", "KDA"]])

print("\nAverage gold per team:")
print(avg_gold)

print("\nAverage damage taken per team:")
print(avg_damage_taken)

print("\nMost used items:")
print(most_used_items)

# Simple heatmap (positions)
plt.figure(figsize=(6, 4))
sns.countplot(x="position", hue="team", data=df)
plt.title("Heatmap simples de posições dos jogadores")
plt.show()