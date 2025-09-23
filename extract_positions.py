import os
import json
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Interactive toggle
SHOW_PLOTS = True
# Archive older outputs instead of deleting (keep off to leave all versions in figures/E3)
ARCHIVE_OLD_OUTPUTS = False

# Load environment and API key (from .env)
load_dotenv(override=True)
API_KEY = os.getenv("LEAGUE_API_KEY")
if not API_KEY:
    print("LEAGUE_API_KEY not found in environment. Set it in .env and retry.")
    raise SystemExit(1)

# Seaborn theme
sns.set_theme(style="white", context="talk", palette="viridis")
# Prefer fonts that support CJK to avoid missing glyph warnings (Windows first)
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "DejaVu Sans", "Arial"
]
plt.rcParams["axes.unicode_minus"] = False

# Summoner's Rift coordinate system and tuning
MAP_MIN, MAP_MAX = 0, 14820  # Riot timeline coordinates (approximately square)
GRID_BINS = 12               # fewer, larger bins to reduce gaps (tweak as needed)
SMOOTH_KERNEL = 5            # stronger smoothing to fill sparse gaps (odd integer >= 1)
MASK_MARGIN = 2000           # hide outside-corner triangles (approximate SR diamond)

# Figure saving setup
FIG_ROOT = Path("figures")
(FIG_ROOT / "E3").mkdir(parents=True, exist_ok=True)
# Use timestamp to avoid overwriting within the same day
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

# Archive helper
def archive_previous_outputs():
    e3_dir = FIG_ROOT / "E3"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = e3_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "E3_script_*_Heatmap.png",
        "E3_script_*_GridHeatmap_Annotated.png",
    ]
    moved = 0
    for pattern in patterns:
        for p in e3_dir.glob(pattern):
            try:
                p.rename(archive_dir / p.name)
                moved += 1
            except Exception as e:
                print(f"Warning: could not archive {p}: {e}")
    print(f"Archived {moved} previous file(s) to {archive_dir}")

def save_fig(fig, filename: str, subdir: str = None, dpi: int = 220):
    out_dir = FIG_ROOT if not subdir else FIG_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path.as_posix(), dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {out_path}")
    return out_path.as_posix()

# Load your match info file
with open("arutnevjr_ajr_matches_info.json", "r", encoding="utf-8") as f:
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

# Simple 2D smoothing (uniform mean filter)
def _smooth2d(arr: np.ndarray, k: int = 3) -> np.ndarray:
    k = int(max(1, k))
    if k % 2 == 0:
        k += 1  # enforce odd
    if k <= 1:
        return arr.astype(float)
    pad = k // 2
    kernel = np.ones((k, k), dtype=float) / (k * k)
    padded = np.pad(arr, pad, mode='edge')
    out = np.zeros_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i:i + k, j:j + k]
            out[i, j] = np.sum(window * kernel)
    return out

# Build a mask to hide outside-corner triangles (approximate SR diamond)
def _sr_corner_mask(bins: int, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])  # length bins
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])  # length bins
    # After transpose, heat[y, x] -> meshgrid(y first, then x)
    YY, XX = np.meshgrid(ycenters, xcenters, indexing='ij')  # shape (bins, bins)
    # Hide two opposite corners outside the diagonal playable area
    min_sum = MAP_MIN + MASK_MARGIN
    max_sum = MAP_MAX * 2 - MASK_MARGIN
    mask = (XX + YY < min_sum) | (XX + YY > max_sum)
    return mask

# Helper to build annotated grid heatmap from x,y coordinates
def plot_annotated_grid_heatmap(player_df: pd.DataFrame, riotid: str, bins: int = GRID_BINS):
    # Clamp to map bounds
    x = player_df["x"].clip(MAP_MIN, MAP_MAX).to_numpy()
    y = player_df["y"].clip(MAP_MIN, MAP_MAX).to_numpy()

    # Bin positions into fixed SR coordinate space
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[bins, bins],
        range=[[MAP_MIN, MAP_MAX], [MAP_MIN, MAP_MAX]]
    )

    # Rows as Y, Cols as X for seaborn heatmap
    heat_raw = H.T.astype(int)

    # Smoothed copy for color fill to reduce "holes"; keep raw counts for annotations
    heat_smooth = _smooth2d(heat_raw, SMOOTH_KERNEL)

    # Build annotations: empty string for zeros to reduce clutter
    annot = heat_raw.astype(str)
    annot[heat_raw == 0] = ""

    # SR corner mask to hide invalid corners
    mask = _sr_corner_mask(bins, xedges, yedges)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        heat_smooth,
        ax=ax,
        cmap="mako",
        cbar=True,
        annot=annot,
        fmt="",
        linewidths=0.6,
        linecolor="white",
        square=True,
        mask=mask,
        cbar_kws={"label": "Position intensity (smoothed)"}
    )

    # Put origin at bottom to match LoL coordinates
    ax.invert_yaxis()

    # Label ticks with map coords (approximate, evenly spaced)
    step = max(1, bins // 5)
    xticks = np.arange(0.5, bins, step)
    yticks = np.arange(0.5, bins, step)
    xlabels = [f"{int(v):d}" for v in np.linspace(MAP_MIN, MAP_MAX, bins)[::step]]
    ylabels = [f"{int(v):d}" for v in np.linspace(MAP_MIN, MAP_MAX, bins)[::step]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, rotation=0)

    ax.set_title(f"Annotated Grid Heatmap: {riotid}", fontsize=15, weight='bold')
    ax.set_xlabel("X coordinate (~SR units)")
    ax.set_ylabel("Y coordinate (~SR units)")
    fig.tight_layout()

    safe_name = riotid.replace('#', '_').replace(' ', '_')
    save_fig(fig, f"E3_script_{RUN_STAMP}_{safe_name}_GridHeatmap_Annotated.png", subdir="E3")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# Generate visuals per player (annotated grid only)
for pid in sorted(df["participantId"].unique(), key=int):
    player_df = df[df["participantId"] == pid].sort_values("timestamp")
    puuid = participantid_to_puuid.get(str(pid))
    riotid = puuid_to_riotid.get(puuid, f"Player {pid}")

    # Annotated spreadsheet-style heatmap only
    plot_annotated_grid_heatmap(player_df, riotid, bins=GRID_BINS)
