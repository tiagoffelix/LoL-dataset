import requests
import os
import json
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

load_dotenv()
API_KEY = os.getenv("LEAGUE_API_KEY")
print("Loaded API KEY:", API_KEY)  # Debug print

# Load your match info file
with open("Speazyy_EUW_matches_info.json", "r") as f:
    matches_info = json.load(f)

# Get the first match id (or any match you want)
match_id = matches_info[0]["matches"][0]

# Fetch timeline data for the match
API_KEY = os.getenv("LEAGUE_API_KEY")
REGION_URL = 'europe.api.riotgames.com/lol/'
MATCH_INFORMATION_ENDPOINT = 'match/v5/matches/'
timeline_url = f"https://{REGION_URL}{MATCH_INFORMATION_ENDPOINT}{match_id}/timeline"
headers = {"X-Riot-Token": API_KEY}

timeline = requests.get(timeline_url, headers=headers).json()

# Check for 'info' key before processing
if "info" not in timeline:
    print("Timeline API response did not contain 'info'. Full response:")
    print(json.dumps(timeline, indent=2))
    exit(1)

# Extract positions for all players across all frames
positions = []
for frame in timeline["info"]["frames"]:
    for pid, pdata in frame["participantFrames"].items():
        if "position" in pdata:
            positions.append({
                "participantId": pid,
                "x": pdata["position"]["x"],
                "y": pdata["position"]["y"],
                "timestamp": frame["timestamp"]
            })

df = pd.DataFrame(positions)


# Load map image
map_img = mpimg.imread("lol_map.png")

# Plot movement path for each player
for pid in sorted(df["participantId"].unique(), key=int):
    player_df = df[df["participantId"] == pid].sort_values("timestamp")
    plt.figure(figsize=(8, 6))
    # Show map image as background, fit to 0-18000
    plt.imshow(map_img, extent=[0, 18000, 0, 18000], aspect='auto', alpha=0.6)
    # KDE heatmap for density
    sns.kdeplot(x=player_df["x"], y=player_df["y"], fill=True, cmap="viridis", bw_adjust=0.5, alpha=0.7)
    # Overlay movement path
    plt.plot(player_df["x"], player_df["y"], color="red", marker="o", markersize=2, linewidth=1, alpha=0.8, label="Path")
    plt.title(f"Movement Heatmap & Path for Player {pid}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 18000)
    plt.ylim(0, 18000)
    plt.legend()
    plt.grid(True)
    plt.show()