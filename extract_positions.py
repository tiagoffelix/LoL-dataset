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

# Interactive toggle
SHOW_PLOTS = True

# Load environment and API key (from .env)
load_dotenv(override=True)
API_KEY = os.getenv("LEAGUE_API_KEY")
if not API_KEY:
    print("LEAGUE_API_KEY not found in environment. Set it in .env and retry.")
    raise SystemExit(1)

# Seaborn theme
sns.set_theme(style="dark", context="notebook", palette="viridis")

# Figure saving setup
FIG_ROOT = Path("figures")
(FIG_ROOT / "E3").mkdir(parents=True, exist_ok=True)
RUN_DATE = datetime.now(timezone.utc).date().isoformat()

def save_fig(fig, filename: str, subdir: str = None, dpi: int = 220):
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

# Build mapping from puuid to Riot ID
puuid_to_riotid = {}
for participant in matchinfo["info"]["participants"]:
    puuid = participant.get("puuid")
    name = participant.get("riotIdGameName")
    tag = participant.get("riotIdTagline")
    if not name or not tag:
        name = participant.get("summonerName")
        tag = ""
    if puuid and name:
        puuid_to_riotid[puuid] = f"{name}#{tag}" if tag else name

print("Player names in this match:")
for puuid, riotid in puuid_to_riotid.items():
    print(riotid)

participantid_to_puuid = {}
for participant in matchinfo["info"]["participants"]:
    pid = str(participant.get("participantId"))
    puuid = participant.get("puuid")
    if pid and puuid:
        participantid_to_puuid[pid] = puuid

# Enhanced heatmap + trajectory per player
for pid in sorted(df["participantId"].unique(), key=int):
    player_df = df[df["participantId"] == pid].sort_values("timestamp")
    puuid = participantid_to_puuid.get(str(pid))
    riotid = puuid_to_riotid.get(puuid, f"Player {pid}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(map_img, extent=[0, 18000, 0, 18000], aspect='auto', alpha=0.5)

    # Kernel density heatmap
    sns.kdeplot(x=player_df["x"], y=player_df["y"], fill=True, thresh=0.05, levels=50,
                cmap="viridis", bw_adjust=0.6, alpha=0.85, ax=ax)

    # Trajectory line (smoothed appearance via alpha)
    ax.plot(player_df["x"], player_df["y"], color="#ff6f69", linewidth=1.2, alpha=0.9, label="Path")
    ax.scatter(player_df["x"].iloc[0], player_df["y"].iloc[0], color="#2ecc71", s=40, zorder=5, label="Start")
    ax.scatter(player_df["x"].iloc[-1], player_df["y"].iloc[-1], color="#e74c3c", s=40, zorder=5, label="End")

    ax.set_title(f"Movement Density & Path: {riotid}", fontsize=14, weight='bold')
    ax.set_xlim(0, 18000)
    ax.set_ylim(0, 18000)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(frameon=True)
    ax.grid(alpha=0.15, linestyle='--')
    for spine in ax.spines.values():
        spine.set_alpha(0.3)

    safe_name = riotid.replace('#', '_').replace(' ', '_')
    save_fig(fig, f"E3_script_{RUN_DATE}_{safe_name}_Heatmap.png", subdir="E3")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
