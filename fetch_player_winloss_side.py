from webbrowser import get
import requests
from dotenv import load_dotenv
import json
import os
import time


load_dotenv()

# Set your Riot API key and player info here
API_KEY = os.getenv("LEAGUE_API_KEY")
REGION = "europe"  
PLATFORM = "europe" 
SUMMONER_NAME = "arutnevjr"  
TAGLINE = "ajr"  

headers = {"X-Riot-Token": API_KEY}

def get_puuid(summoner_name, tagline):
    url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{summoner_name}/{tagline}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["puuid"]

def get_match_ids(puuid, count=100, start=0):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()

def get_match_info(match_id):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()

def extract_winloss_side(matchinfo, puuid):
    for participant in matchinfo["info"]["participants"]:
        if participant["puuid"] == puuid:
            win = participant["win"]
            team_id = participant["teamId"]
            side = "blue" if team_id == 100 else "red"
            return {"win": win, "side": side}
    return None

def main():
    puuid = get_puuid(SUMMONER_NAME, TAGLINE)

    all_results = []
    # Only fetch the last 10 matches
    match_ids = get_match_ids(puuid, count=100, start=0)
    time.sleep(0.1)  # Small delay for safety

    for match_id in match_ids:
        try:
            matchinfo = get_match_info(match_id)
            result = extract_winloss_side(matchinfo, puuid)
            if result:
                all_results.append(result)
            time.sleep(0.1)  # Increased delay to reduce 429 errors
        except Exception as e:
            print(f"Error for match {match_id}: {e}")

    output = {
        "summoner_name": SUMMONER_NAME,
        "tagline": TAGLINE,
        "results": all_results
    }
    with open("player_winloss_side.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
