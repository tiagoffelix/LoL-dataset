from webbrowser import get
import requests
from dotenv import load_dotenv
import json
import os
import time
from lookup import id_to_name
from helper import fill_json
from helper import get_ten_min_gold
from helper import concat_json
from summoner_list import summoner_list

load_dotenv()


API_URL = 'euw1.api.riotgames.com/lol/'
API_KEY = os.getenv("LEAGUE_API_KEY")
REGION_URL = 'europe.api.riotgames.com/lol/'
MATCH_ID_ENDPOINT = 'match/v5/matches/by-puuid/'
MATCH_INFORMATION_ENDPOINT = 'match/v5/matches/'
SUMMONER_BY_NAME = 'summoner/v4/summoners/by-name/'
num_requests = 0
total_requests = 0

print(os.getenv("LEAGUE_API_KEY"))

# Returns puuid (str) for a given Riot ID (name/tag)
def get_puuid(riot_name, riot_tag):
	# riot_name: e.g. 'Speazyy', riot_tag: e.g. 'EUW'
	get_url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{riot_name}/{riot_tag}"
	headers = {"X-Riot-Token": API_KEY}
	r = requests.get(get_url, headers=headers)
	if r.status_code == requests.codes.ok:
		data = r.json()
		return data['puuid']
	else:
		print(f"Failed to get puuid for {riot_name}#{riot_tag}")
		print(f"Error code: {r.status_code}")
		print(f"Response body: {r.text}")
		return None

""" Helper functions that keep track of # of requests """
# Check to ensure we didn't exceed the API request limit for every 2 min
def reached_request_limit():
	global num_requests
	if num_requests >= 98:
		print(f"Made {num_requests} requests. Sleep for 2 min.")
		global total_requests
		total_requests += num_requests
		num_requests = 0
		
		return True
	else:
		return False

def make_and_verify_request(get_url):
	r = requests.get(get_url, 
		headers={"X-Riot-Token": API_KEY})
	
	global num_requests
	num_requests += 1

	# Verify
	if r.status_code == requests.codes.ok:
		return r.json()

	else:
		print(f"Something happened with the request {get_url}")
		print(f"Error code: {r.status_code}")
		print(f"Response body: {r.json}")
		return ""

"""
get 20 match ids for each summoner
	params:
		puuid: uniquely identifies summoner
	returns:
		a list of ids of the last 20 ranks games played
"""
def get_matches(puuid):

	if reached_request_limit():
		time.sleep(120)
		print("resume from get_matches()")

	get_url = 'https://' + REGION_URL + MATCH_ID_ENDPOINT
	match_type = {'type':'ranked', 'start':'0', 'count':'1'} 
		
	r = requests.get(get_url + puuid + '/ids', 
				headers={"X-Riot-Token": API_KEY},
				params=match_type)

	# Verify
	if r.status_code == requests.codes.ok:
		data = r.json()
	else:
		print(f"Something happened with the request {get_url}")
		print(f"Error code: {r.status_code}")
		print(f"Response body: {r.json}")
		data = ""

	global num_requests
	num_requests += 1

	return data

# query timeline api for 10 min gold data
def fetch_timeline_info(match_id, position):
	if reached_request_limit():
		time.sleep(120)
		print("resume from fetch_timeline_info")

	get_url = 'https://' + REGION_URL + MATCH_INFORMATION_ENDPOINT + match_id + '/timeline'
	data = make_and_verify_request(get_url) # use requests.get()

	# if request failed
	if data == "":
		exit()
	ten_min_gold = get_ten_min_gold(data['info']['frames'])
	opponent_ten_min_gold = [0] * 10 

	for index in range(0,5):
		for jndex in range(5,10):
			if position[index] == position[jndex]: 
				opponent_ten_min_gold[index] = ten_min_gold[jndex]
				opponent_ten_min_gold[jndex] = ten_min_gold[index]
				break;
	return ten_min_gold, opponent_ten_min_gold


"""
get details of each match
	params:
		match ID: uniquely identifies a game
	returns:
		a dictionary that contains picks, bans, wins, kda, and gold earned
"""

def get_match_info(match_id):
	if reached_request_limit():
		time.sleep(120)
		print("resume from get_match_info")

	get_url = 'https://' + REGION_URL + MATCH_INFORMATION_ENDPOINT + match_id
	game_data = make_and_verify_request(get_url)

	# Prepare lists for each stat
	matches, picks, position, bans, teams = [], [], [], [], []
	kills, deaths, assists, gold, win = [], [], [], [], []
	damage_dealt, damage_taken, items = [], [], []
	riot_game_names, riot_taglines = [], []
	kda = []

	index = 0
	for player in game_data["info"]["participants"]:
		matches.append(match_id)
		picks.append(player.get("championName"))
		position.append(player.get('individualPosition'))
		kills.append(player.get("kills"))
		deaths.append(player.get("deaths"))
		assists.append(player.get("assists"))
		gold.append(player.get("goldEarned"))
		win.append(player.get("win"))
		damage_dealt.append(player.get("totalDamageDealtToChampions"))
		damage_taken.append(player.get("totalDamageTaken"))
		# KDA calculation
		d = player.get("deaths")
		kda.append((player.get("kills") + player.get("assists")) / d if d != 0 else player.get("kills") + player.get("assists"))
		# Items
		items.append([player.get(f"item{i}") for i in range(7)])
		# Riot ID gameName and tagline
		puuid = player.get("puuid")
		riot_id_url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
		headers = {"X-Riot-Token": API_KEY}
		riot_id_resp = requests.get(riot_id_url, headers=headers)
		if riot_id_resp.status_code == requests.codes.ok:
			riot_id_data = riot_id_resp.json()
			riot_game_names.append(riot_id_data.get("gameName", ""))
			riot_taglines.append(riot_id_data.get("tagLine", ""))
		else:
			riot_game_names.append("")
			riot_taglines.append("")

		# Team and bans
		if index < 5:
			champion_id = game_data["info"]["teams"][0]["bans"][index]["championId"]
			bans.append(id_to_name(champion_id))
			teams.append("blue")
		else:
			champion_id = game_data["info"]["teams"][1]["bans"][index % 5]["championId"]
			bans.append(id_to_name(champion_id))
			teams.append("red")
		index += 1

	ten_min_gold, opponent_ten_min_gold = fetch_timeline_info(match_id, position)

	return {
		"matches": matches,
		"picks": picks,
		"position": position,
		"riotIdGameName": riot_game_names,
		"riotIdTagline": riot_taglines,
		"bans": bans,
		"teams": teams,
		"kills": kills,
		"deaths": deaths,
		"assists": assists,
		"gold": gold,
		"win": win,
		"damageDealt": damage_dealt,
		"damageTaken": damage_taken,
		"items": items,
		"KDA": kda,
		"tenMinGold": ten_min_gold,
		"tenMinLaneOpponentGold": opponent_ten_min_gold,
	}

# print(get_match_info('NA1_4250997412'))
# s = ['-oXMqiG7Iz4jfAcbhR09AeP44KvBCtL9cEejVh-adG5LlQ0PEQFLJSwJV0Xk7upjLNKm5l3fygLhWA']
"""
generate 5000 rows of game data from:
	25 (summoners) x 20 (of their ranked games) x 10 (for the # of players per game)
	returns: writes to the "lots_and_lots_of_data.json"
"""
def populate_dataset(summoner_list):
	data = {}
	responses = []
	for index, summoner in enumerate(summoner_list):
		print(f"Adding {index}/{len(summoner_list)} summoner data!")
		matches = get_matches(summoner)
		for match in matches:
			responses.append(get_match_info(match))

		# to not exceed Riot Games' API request rate limit
		# needs 16 min and 40 seconds to finish
		print(f"Finished {index+1}/{len(summoner_list)}")
		global num_requests
		print(f"Made {num_requests} requests.")
	
	for response in responses:
		data = concat_json(data, response)

	with open('lots_and_lots_of_data.json', 'w') as s:
		s.write(json.dumps(data, indent = 4))


# Utility: Print last 20 matches for a given summoner name
if __name__ == "__main__":
	riot_name = 'arutnevjr'
	riot_tag = 'ajr'
	puuid = get_puuid(riot_name, riot_tag)
	print(f"PUUID for {riot_name}#{riot_tag}: {puuid}")
	if puuid:
		matches = get_matches(puuid)
		print(f"Last 20 matches for {riot_name}#{riot_tag}:\n{matches}")
		print("\nFetching match info for each match...")
		all_match_info = []
		for match_id in matches:
			info = get_match_info(match_id)
			all_match_info.append(info)
			print(f"Match {match_id} info:\n{json.dumps(info, indent=2)}\n")
		# Optionally, save all info to a file
		with open(f"{riot_name}_{riot_tag}_matches_info.json", "w") as f:
			json.dump(all_match_info, f, indent=2)
		print(f"All match info saved to {riot_name}_{riot_tag}_matches_info.json")
	else:
		print("Could not retrieve matches.")