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

# Returns puuid (str) for a given summoner name
def get_puuid(summoner_name):
	get_url = 'https://' + API_URL + SUMMONER_BY_NAME
	r = requests.get(get_url + summoner_name, headers={"X-Riot-Token": API_KEY})
	if r.status_code == requests.codes.ok:
		data = r.json()
		return data['puuid']
	else:
		print(f"Failed to get puuid for {summoner_name}")
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
	match_type = {'type':'ranked', 'start':'0', 'count':'20'} 
		
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

	# make empty cols
	matches = []
	picks = []
	position = []
	bans = []
	teams = []
	kills = []
	deaths = []
	assists = []
	gold = []
	challenges = []
	win = []

	# to keep track of 10 players in 1 game
	index = 0
	# populate each row with a player's data
	for player in game_data["info"]["participants"]:
		row = {}
		# Ids
		matches.append(match_id) 
		picks.append(player.get("championName"))

		position.append(player.get('individualPosition'))
		# there are 2 teams
		if index < 5:
			champion_id = game_data["info"]["teams"][0]["bans"][index]["championId"]
			bans.append(id_to_name(champion_id))
			teams.append("blue")
		else:
			# mod index so index of 2nd team starts at 0 too
			champion_id = game_data["info"]["teams"][1]["bans"][index%5]["championId"]
			bans.append(id_to_name(champion_id))
			teams.append("red")

		kills.append(player.get("kills"))
		deaths.append(player.get("deaths"))
		assists.append(player.get("assists"))
		gold.append(player.get("goldEarned"))
		challenges.append(fill_json(player.get("challenges", {})))
		win.append(player.get("win"))

		index += 1
	
	# query timeline API to get gold @ 10 min
	ten_min_gold, opponent_ten_min_gold = fetch_timeline_info(match_id, position)

	return {
		"matches": matches,
		"picks": picks,
		"position": position,
		"bans": bans,
		"teams": teams,
		"kills": kills,
		"deaths": deaths,
		"assists": assists,
		"gold": gold,
		"tenMinGold": ten_min_gold,
		"tenMinLaneOpponentGold": opponent_ten_min_gold,
		"challenges": challenges,
		"win": win
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
	summoner_name = 'Speazyy'
	puuid = get_puuid(summoner_name)
	if puuid:
		matches = get_matches(puuid)
		print(f"Last 20 matches for {summoner_name}:\n{matches}")
	else:
		print("Could not retrieve matches.")