import json
import matplotlib.pyplot as plt

with open("player_winloss_side.json", "r") as f:
    json_data = json.load(f)

SUMMONER_NAME = json_data.get("summoner_name", "?")
TAGLINE = json_data.get("tagline", "?")
data = json_data.get("results", [])

counts = {
    "Blue Win": 0,
    "Blue Loss": 0,
    "Red Win": 0,
    "Red Loss": 0
}

for entry in data:
    if entry["side"] == "blue":
        if entry["win"]:
            counts["Blue Win"] += 1
        else:
            counts["Blue Loss"] += 1
    else:
        if entry["win"]:
            counts["Red Win"] += 1
        else:
            counts["Red Loss"] += 1

labels = [f"{k} ({v})" for k, v in counts.items()]
sizes = list(counts.values())
colors = ["#4A90E2", "#B0C4DE", "#E94B3C", "#F7CAC9"]

# Calculate totals and winrate
total_wins = counts["Blue Win"] + counts["Red Win"]
total_losses = counts["Blue Loss"] + counts["Red Loss"]
total_games = total_wins + total_losses
winrate = (total_wins / total_games * 100) if total_games > 0 else 0

plt.figure(figsize=(7,7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, counterclock=False)
plt.title(f"Win/Loss Rate by Side for {SUMMONER_NAME}#{TAGLINE}\nTotal games: {total_games} | Wins: {total_wins} | Losses: {total_losses} | Winrate: {winrate:.1f}%")
plt.axis('equal')
plt.show()
