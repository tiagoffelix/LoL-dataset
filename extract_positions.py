import requests
import os
import json
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from pathlib import Path
from datetime import datetime, timezone

# Load environment and API key (from .env)
load_dotenv(override=True)
API_KEY = os.getenv("LEAGUE_API_KEY")
if not API_KEY:
    print("LEAGUE_API_KEY not found in environment. Set it in .env and retry.")
    raise SystemExit(1)

# Figure saving setup
FIG_ROOT = Path("figures")
(FIG_ROOT / "E3").mkdir(parents=True, exist_ok=True)
RUN_DATE = datetime.now(timezone.utc).date().isoformat()

def save_fig(fig, filename: str, subdir: str = None, dpi: int = 200):
    out_dir = FIG_ROOT if not subdir else FIG_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path.as_posix(), dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {out_path}")
    return out_path.as_posix()

# Load your match info file
with open("arutnevjr_ajr_matches_info.json", "r") as f:
    matches_info = json.load(f)

# Get the first match id (or any match you want)
match_id = matches_info[0]["matches"][0]


# Prepare endpoints
REGION_URL = 'europe.api.riotgames.com/lol/'
MATCH_INFORMATION_ENDPOINT = 'match/v5/matches/'
headers = {"X-Riot-Token": API_KEY}

timeline_url = f"https://{REGION_URL}{MATCH_INFORMATION_ENDPOINT}{match_id}/timeline"
resp_tl = requests.get(timeline_url, headers=headers)
if resp_tl.status_code != 200:
    print(f"Timeline request failed: {resp_tl.status_code}\n{resp_tl.text}")
    raise SystemExit(1)

timeline = resp_tl.json()

# Fetch match info for player names
matchinfo_url = f"https://{REGION_URL}{MATCH_INFORMATION_ENDPOINT}{match_id}"
resp_mi = requests.get(matchinfo_url, headers=headers)
if resp_mi.status_code != 200:
    print(f"Match info request failed: {resp_mi.status_code}\n{resp_mi.text}")
    raise SystemExit(1)

matchinfo = resp_mi.json()

# Check for 'info' key before processing
if "info" not in timeline:
    print("Timeline API response did not contain 'info'. Full response:")
    print(json.dumps(timeline, indent=2))
    raise SystemExit(1)

# Extract positions for all players across all frames
positions = []
for frame in timeline["info"]["frames"]:
    for pid, pdata in frame["participantFrames"].items():
        if "position" in pdata:
            positions.append({
                "participantId": str(pid),
                "x": pdata["position"]["x"],
                "y": pdata["position"]["y"],
                "timestamp": frame["timestamp"]
            })

df = pd.DataFrame(positions)

# Load map image
map_img = mpimg.imread("lol_map.png")





## Build a mapping from puuid to Riot ID (name#tag) using match info
puuid_to_riotid = {}
for participant in matchinfo["info"]["participants"]:
    puuid = participant.get("puuid")
    name = participant.get("riotIdGameName")
    tag = participant.get("riotIdTagline")
    # fallback to summonerName if riotIdGameName/tag missing
    if not name or not tag:
        name = participant.get("summonerName")
        tag = ""
    if puuid and name:
        puuid_to_riotid[puuid] = f"{name}#{tag}" if tag else name

# Print out the player names (Riot ID)
print("Player names in this match:")
for puuid, riotid in puuid_to_riotid.items():
    print(riotid)

## Build a mapping from participantId to puuid using match info participants
participantid_to_puuid = {}
for participant in matchinfo["info"]["participants"]:
    pid = str(participant.get("participantId"))
    puuid = participant.get("puuid")
    if pid and puuid:
        participantid_to_puuid[pid] = puuid

# Plot movement path for each player
for pid in sorted(df["participantId"].unique(), key=int):
    player_df = df[df["participantId"] == pid].sort_values("timestamp")
    puuid = participantid_to_puuid.get(str(pid))
    riotid = puuid_to_riotid.get(puuid, f"Player {pid}")
    print(f"DEBUG: pid={pid}, puuid={puuid}, riotid={riotid}")
    plt.figure(figsize=(8, 6))
    plt.imshow(map_img, extent=[0, 18000, 0, 18000], aspect='auto', alpha=0.6)
    sns.kdeplot(x=player_df["x"], y=player_df["y"], fill=True, cmap="viridis", bw_adjust=0.5, alpha=0.7)
    plt.plot(player_df["x"], player_df["y"], color="red", marker="o", markersize=2, linewidth=1, alpha=0.8, label="Path")


    plt.title(f"Movement Heatmap & Path for {riotid}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 18000)
    plt.ylim(0, 18000)
    plt.legend()
    plt.grid(True)
    # Save each player's heatmap
    safe_name = riotid.replace('#', '_').replace(' ', '_')
    save_fig(plt.gcf(), f"E3_script_{RUN_DATE}_{safe_name}_Heatmap.png", subdir="E3")
    plt.show()
