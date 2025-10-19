import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from lookup import id_to_name
from summoner_list import summoner_list


load_dotenv()

# Map platform prefix to regional routing cluster for Match-V5
_MATCH_REGION_BY_PREFIX = {
    # Europe
    "EUW1": "https://europe.api.riotgames.com",
    "EUN1": "https://europe.api.riotgames.com",
    "TR1": "https://europe.api.riotgames.com",
    "RU": "https://europe.api.riotgames.com",
    # Americas
    "NA1": "https://americas.api.riotgames.com",
    "BR1": "https://americas.api.riotgames.com",
    "LA1": "https://americas.api.riotgames.com",
    "LA2": "https://americas.api.riotgames.com",
    "OC1": "https://americas.api.riotgames.com",
    # Asia
    "KR": "https://asia.api.riotgames.com",
    "JP1": "https://asia.api.riotgames.com",
    # SEA
    "PH2": "https://sea.api.riotgames.com",
    "SG2": "https://sea.api.riotgames.com",
    "TH2": "https://sea.api.riotgames.com",
    "TW2": "https://sea.api.riotgames.com",
    "VN2": "https://sea.api.riotgames.com",
}


def _normalize_champion_name(name: str) -> str:
    """Normalize champion names for reliable comparisons."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _short_patch(version: str) -> str:
    if not version:
        return version
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def _parse_csv(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return tuple()
    items = []
    for fragment in value.split(","):
        token = fragment.strip()
        if token:
            items.append(token)
    return tuple(items)


def _parse_queue_ids(raw: Optional[str]) -> Tuple[int, ...]:
    if not raw:
        return (420, 440)
    values = []
    for fragment in raw.split(","):
        fragment = fragment.strip()
        if not fragment:
            continue
        try:
            values.append(int(fragment))
        except ValueError:
            continue
    return tuple(values) if values else (420, 440)


def _fetch_patch_versions(limit: int = 5) -> List[str]:
    """Fetch the latest live patches from Data Dragon."""
    try:
        response = requests.get(
            "https://ddragon.leagueoflegends.com/api/versions.json", timeout=5
        )
        response.raise_for_status()
        versions = response.json()
        if isinstance(versions, list):
            return [str(v) for v in versions[:limit]]
    except Exception as exc:
        print(f"[warn] Failed to fetch patch versions from Data Dragon: {exc}")
    return []


def _infer_default_patches() -> Dict[str, str]:
    versions = _fetch_patch_versions(limit=4)
    if versions:
        current = _short_patch(versions[0])
        previous = _short_patch(versions[1]) if len(versions) > 1 else current
        print(
            f"[*] Auto-detected patches: current={current}, previous={previous} "
            "(override via CURRENT_PATCH/PREVIOUS_PATCH if needed)."
        )
        return {"current": current, "previous": previous}
    try:
        print("[warn] Using fallback patches (15.20 / 15.19); network detection failed.")
    except Exception:
        pass
    return {"current": "15.20", "previous": "15.19"}


@dataclass
class ChampionBalanceConfig:
    champion_name: str = "Volibear"
    patches: Dict[str, str] = field(
        default_factory=_infer_default_patches
    )
    matches_per_patch: int = 60
    match_search_timeout_seconds: int = 1200
    queues: Tuple[int, ...] = (420, 440)
    max_matches_per_player: int = 400
    matches_batch_size: int = 75
    output_path: str = "champion_balance_stats.json"
    store_match_ids: bool = True
    match_type: Optional[str] = "ranked"  # filter for match-v5 ids endpoint
    debug_requests: bool = False           # verbose logging of API requests
    region_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "https://europe.api.riotgames.com",
            "https://americas.api.riotgames.com",
            "https://asia.api.riotgames.com",
            "https://sea.api.riotgames.com",
        )
    )
    seed_strategy: str = "auto"  # auto -> env overrides -> challenger/ladder fallback
    seed_league_queue: str = "RANKED_SOLO_5x5"
    seed_league_tiers: Tuple[str, ...] = field(
        default_factory=lambda: (
            "PLATINUM",
            "EMERALD",
            "DIAMOND",
            "MASTER",
            "GRANDMASTER",
            "CHALLENGER",
        )
    )
    seed_league_divisions: Tuple[str, ...] = field(
        default_factory=lambda: ("I", "II", "III", "IV")
    )
    seed_entries_per_tier: int = 30
    player_puuids_override: Tuple[str, ...] = field(default_factory=tuple)
    player_puuid_file: Optional[str] = None
    summary_output_path: str = "champion_balance_summary.txt"
    generate_plot: bool = True
    plot_output_path: Optional[str] = None

    def __post_init__(self):
        if self.matches_batch_size > 100:
            raise ValueError(
                "matches_batch_size must not exceed 100 (Riot API limitation)."
            )
        if self.matches_per_patch <= 0:
            raise ValueError("matches_per_patch must be a positive integer.")
        if not self.patches:
            raise ValueError("Configuration must include at least one patch to study.")
        canonical = {}
        for alias, patch in self.patches.items():
            if not patch:
                raise ValueError(f"Patch label for '{alias}' cannot be empty.")
            canonical[alias] = str(patch).strip()
        self.patches = canonical
        if not self.queues:
            self.queues = (420, 440)
        # sanitize match_type
        if self.match_type:
            self.match_type = str(self.match_type).strip().lower()
            if self.match_type not in {"ranked", "normal", "tourney", "tutorial"}:
                self.match_type = "ranked"
        self.seed_strategy = (self.seed_strategy or "auto").strip().lower()
        if self.seed_strategy not in {"auto", "league", "static"}:
            self.seed_strategy = "auto"
        queue_token = (self.seed_league_queue or "RANKED_SOLO_5x5").strip()
        if not queue_token:
            queue_token = "RANKED_SOLO_5x5"
        queue_lookup = {
            "RANKED_SOLO_5X5": "RANKED_SOLO_5x5",
            "RANKED_SOLO_5x5": "RANKED_SOLO_5x5",
            "RANKED_FLEX_SR": "RANKED_FLEX_SR",
            "RANKED_FLEX_TT": "RANKED_FLEX_TT",
        }
        self.seed_league_queue = queue_lookup.get(queue_token.upper(), queue_token)
        if self.seed_entries_per_tier <= 0:
            raise ValueError("seed_entries_per_tier must be positive.")
        tiers = []
        for tier in self.seed_league_tiers:
            tier = str(tier).strip().upper()
            if tier:
                tiers.append(tier)
        self.seed_league_tiers = tuple(tiers) or ("EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER")
        divisions = []
        for div in self.seed_league_divisions:
            div = str(div).strip().upper()
            if div:
                divisions.append(div)
        self.seed_league_divisions = tuple(divisions) or ("I", "II", "III", "IV")
        overrides = []
        for puuid in self.player_puuids_override:
            token = str(puuid).strip()
            if token:
                overrides.append(token)
        self.player_puuids_override = tuple(overrides)
        if self.player_puuid_file:
            self.player_puuid_file = str(self.player_puuid_file).strip() or None
        self.summary_output_path = (
            str(self.summary_output_path).strip() or "champion_balance_summary.txt"
        )
        self.generate_plot = bool(self.generate_plot)
        if self.plot_output_path:
            self.plot_output_path = str(self.plot_output_path).strip() or None


@dataclass
class PatchAccumulator:
    patch: str
    target_matches: int
    store_match_ids: bool
    matches_evaluated: int = 0
    champion_matches: int = 0
    wins: int = 0
    losses: int = 0
    total_kills: int = 0
    total_deaths: int = 0
    total_assists: int = 0
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    total_gold: int = 0
    total_cs: int = 0
    total_vision_score: int = 0
    total_lp: int = 0
    ban_count: int = 0
    queue_distribution: Counter = field(default_factory=Counter)
    role_distribution: Counter = field(default_factory=Counter)
    rank_distribution: Counter = field(default_factory=Counter)
    tier_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    match_ids: List[str] = field(default_factory=list)
    game_versions: Counter = field(default_factory=Counter)

    def record_match(
        self,
        queue_id: Optional[int],
        target_banned: bool,
        game_version: Optional[str],
    ) -> None:
        self.matches_evaluated += 1
        if queue_id is not None:
            self.queue_distribution[str(queue_id)] += 1
        if target_banned:
            self.ban_count += 1
        if game_version:
            self.game_versions[_short_patch(game_version)] += 1

    def record_champion(
        self, match_id: Optional[str], participant: Dict, rank_info: Dict
    ) -> None:
        self.champion_matches += 1
        win = bool(participant.get("win"))
        if win:
            self.wins += 1
        else:
            self.losses += 1

        kills = participant.get("kills", 0)
        deaths = participant.get("deaths", 0)
        assists = participant.get("assists", 0)
        damage_dealt = participant.get("totalDamageDealtToChampions", 0)
        damage_taken = participant.get("totalDamageTaken", 0)
        gold = participant.get("goldEarned", 0)
        cs = participant.get("totalMinionsKilled", 0) + participant.get(
            "neutralMinionsKilled", 0
        )
        vision = participant.get("visionScore", 0)

        self.total_kills += kills
        self.total_deaths += deaths
        self.total_assists += assists
        self.total_damage_dealt += damage_dealt
        self.total_damage_taken += damage_taken
        self.total_gold += gold
        self.total_cs += cs
        self.total_vision_score += vision

        role = participant.get("teamPosition") or participant.get("individualPosition")
        if role:
            self.role_distribution[role] += 1

        tier = "UNRANKED"
        division = ""
        lp = 0
        if rank_info:
            tier = str(rank_info.get("tier", "UNRANKED")).upper() or "UNRANKED"
            division = str(rank_info.get("rank", "")).upper()
            lp = rank_info.get("leaguePoints", 0) or 0
        self.total_lp += lp

        tier_key = tier
        rank_key = f"{tier} {division}".strip()
        self.rank_distribution[rank_key] += 1

        tier_bucket = self.tier_stats.setdefault(
            tier_key,
            {
                "matches": 0,
                "wins": 0,
                "total_kills": 0,
                "total_deaths": 0,
                "total_assists": 0,
                "total_damage_dealt": 0,
                "total_damage_taken": 0,
                "total_gold": 0,
                "total_cs": 0,
                "total_vision_score": 0,
            },
        )

        tier_bucket["matches"] += 1
        if win:
            tier_bucket["wins"] += 1
        tier_bucket["total_kills"] += kills
        tier_bucket["total_deaths"] += deaths
        tier_bucket["total_assists"] += assists
        tier_bucket["total_damage_dealt"] += damage_dealt
        tier_bucket["total_damage_taken"] += damage_taken
        tier_bucket["total_gold"] += gold
        tier_bucket["total_cs"] += cs
        tier_bucket["total_vision_score"] += vision

        if self.store_match_ids and match_id:
            self.match_ids.append(match_id)

    def summarize(self) -> Dict:
        pick_rate = (
            self.champion_matches / self.matches_evaluated
            if self.matches_evaluated
            else None
        )
        ban_rate = (
            self.ban_count / self.matches_evaluated if self.matches_evaluated else None
        )
        win_rate = (
            self.wins / self.champion_matches if self.champion_matches else None
        )
        avg_stats = None
        if self.champion_matches:
            avg_stats = {
                "kills": self.total_kills / self.champion_matches,
                "deaths": self.total_deaths / self.champion_matches,
                "assists": self.total_assists / self.champion_matches,
                "damage_dealt": self.total_damage_dealt / self.champion_matches,
                "damage_taken": self.total_damage_taken / self.champion_matches,
                "gold": self.total_gold / self.champion_matches,
                "cs": self.total_cs / self.champion_matches,
                "vision_score": self.total_vision_score / self.champion_matches,
                "lp": self.total_lp / self.champion_matches,
            }

        tier_breakdown = {}
        for tier, stats in self.tier_stats.items():
            matches = stats["matches"]
            wins = stats["wins"]
            tier_breakdown[tier] = {
                "matches": matches,
                "wins": wins,
                "win_rate": (wins / matches) if matches else None,
                "avg_kills": (stats["total_kills"] / matches) if matches else 0,
                "avg_deaths": (stats["total_deaths"] / matches) if matches else 0,
                "avg_assists": (stats["total_assists"] / matches) if matches else 0,
                "avg_damage_dealt": (stats["total_damage_dealt"] / matches)
                if matches
                else 0,
                "avg_damage_taken": (stats["total_damage_taken"] / matches)
                if matches
                else 0,
                "avg_gold": (stats["total_gold"] / matches) if matches else 0,
                "avg_cs": (stats["total_cs"] / matches) if matches else 0,
                "avg_vision_score": (stats["total_vision_score"] / matches)
                if matches
                else 0,
            }

        return {
            "patch": self.patch,
            "target_matches": self.target_matches,
            "matches_evaluated": self.matches_evaluated,
            "champion_matches": self.champion_matches,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "pick_rate": pick_rate,
            "ban_rate": ban_rate,
            "ban_count": self.ban_count,
            "average_kda": (
                (self.total_kills + self.total_assists) / max(self.total_deaths, 1)
                if self.champion_matches
                else None
            ),
            "average_stats_per_game": avg_stats,
            "queue_distribution": dict(self.queue_distribution),
            "role_distribution": dict(self.role_distribution),
            "rank_distribution": dict(self.rank_distribution),
            "tier_breakdown": tier_breakdown,
            "match_ids": self.match_ids if self.store_match_ids else None,
            "observed_game_versions": dict(self.game_versions),
        }


class RiotAPIClient:
    def __init__(
        self,
        api_key: str,
        region_route: Optional[str] = None,
        platform_route: Optional[str] = None,
        request_timeout: float = 8.0,
        max_retries: int = 4,
        throttle_seconds: float = 1.0,
        debug: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Missing API key. Set LEAGUE_API_KEY in the environment or .env file."
            )
        # If not provided, infer region from platform route prefix
        platform_default = os.getenv("RIOT_PLATFORM_ROUTE", "https://euw1.api.riotgames.com")
        self.platform_route = platform_route or platform_default
        # Try to detect platform prefix (e.g., EUW1, NA1, KR) from platform route
        platform_prefix = None
        try:
            host = self.platform_route.split("//", 1)[1]
            platform_prefix = host.split(".", 1)[0].upper()
        except Exception:
            platform_prefix = None
        inferred_region = _MATCH_REGION_BY_PREFIX.get(platform_prefix or "", None)
        self.region_route = region_route or inferred_region or os.getenv(
            "RIOT_REGION_ROUTE", "https://europe.api.riotgames.com"
        )
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.throttle_seconds = throttle_seconds
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": api_key})
        self._last_request_ts: float = 0.0
        self.debug = debug or os.getenv("DEBUG_RIOT_API", "0").lower() in {"1", "true", "yes"}
        self._puuid_region_cache: Dict[str, str] = {}

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.throttle_seconds:
            time.sleep(self.throttle_seconds - elapsed)

    def _request(
        self, base_url: str, path: str, params: Optional[Dict] = None
    ) -> Optional[Dict]:
        url = f"{base_url}{path}"
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self.session.get(
                    url, params=params, timeout=self.request_timeout
                )
                self._last_request_ts = time.time()
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
                continue

            if self.debug:
                print(f"[debug] GET {response.url} -> {response.status_code}")

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "2"))
                if self.debug:
                    print(f"[debug] 429 retry-after={retry_after}s")
                time.sleep(retry_after + 0.5)
                continue
            if 500 <= response.status_code < 600:
                if self.debug:
                    print(f"[debug] {response.status_code} server error, retrying...")
                time.sleep(2**attempt)
                continue
            if response.status_code == 404:
                return None
            if response.status_code >= 400:
                # Provide detailed diagnostics but do not crash immediately on 400
                msg = response.text[:300]
                raise requests.HTTPError(
                    f"HTTP {response.status_code} for {url} params={params} body={msg}"
                )

            if not response.text:
                return {}
            return response.json()

        raise RuntimeError(f"Failed to fetch '{url}' after {self.max_retries} attempts.")

    def _platform_hosts(self) -> List[str]:
        return [
            "EUW1", "EUN1", "TR1", "RU",
            "NA1", "BR1", "LA1", "LA2", "OC1",
            "KR", "JP1",
            "PH2", "SG2", "TH2", "TW2", "VN2",
        ]

    def _platform_base(self, prefix: str) -> str:
        return f"https://{prefix.lower()}.api.riotgames.com"

    def resolve_region_for_puuid(self, puuid: str) -> Optional[str]:
        if puuid in self._puuid_region_cache:
            return self._puuid_region_cache[puuid]
        for prefix in self._platform_hosts():
            platform_base = self._platform_base(prefix)
            try:
                if self.debug:
                    print(f"[debug] Probe platform {prefix} for puuid")
                data = self._request(
                    platform_base, f"/lol/summoner/v4/summoners/by-puuid/{puuid}"
                )
                if data is None:
                    continue
                region = _MATCH_REGION_BY_PREFIX.get(prefix, None)
                if region:
                    if self.debug:
                        print(f"[debug] puuid mapped to platform {prefix} -> region {region}")
                    self._puuid_region_cache[puuid] = region
                    return region
            except requests.HTTPError as e:
                # 404 is normal when probing wrong shard; ignore others
                if self.debug:
                    print(f"[debug] probe {prefix} failed: {e}")
                continue
        return None

    def get_match_ids(
        self, puuid: str, start: int, count: int, queue: Optional[int] = None, match_type: Optional[str] = None
    ) -> List[str]:
        params = {"start": max(0, int(start)), "count": int(count)}
        if queue is not None:
            params["queue"] = int(queue)
        if match_type:
            params["type"] = match_type
        # Prefer resolved region per PUUID
        region_base = self._puuid_region_cache.get(puuid) or self.resolve_region_for_puuid(puuid) or self.region_route
        try:
            data = self._request(
                region_base, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params
            )
            return data or []
        except requests.HTTPError as e:
            msg = str(e)
            if "Exception decrypting" in msg or "Bad Request" in msg:
                # attempt fallbacks
                clusters = [
                    "https://americas.api.riotgames.com",
                    "https://asia.api.riotgames.com",
                    "https://europe.api.riotgames.com",
                    "https://sea.api.riotgames.com",
                ]
                for cluster in clusters:
                    if self.debug:
                        print(f"[debug] Retrying match IDs on cluster {cluster}")
                    try:
                        data = self._request(
                            cluster, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params
                        )
                        if data is not None:
                            self._puuid_region_cache[puuid] = cluster
                            return data or []
                    except requests.HTTPError as e2:
                        if self.debug:
                            print(f"[debug] Fallback cluster {cluster} failed: {e2}")
                        continue
            raise

    def get_match(self, match_id: str) -> Optional[Dict]:
        # Route by match ID prefix (e.g., EUW1_*, NA1_*)
        try:
            prefix = (match_id or "").split("_", 1)[0].upper()
        except Exception:
            prefix = ""
        base = _MATCH_REGION_BY_PREFIX.get(prefix, self.region_route)
        return self._request(base, f"/lol/match/v5/matches/{match_id}")

    def get_league_entries(self, summoner_id: str) -> List[Dict]:
        data = self._request(
            self.platform_route, f"/lol/league/v4/entries/by-summoner/{summoner_id}"
        )
        return data or []

    def get_league_entries_page(
        self, queue: str, tier: str, division: str, page: int = 1
    ) -> List[Dict]:
        params = {"page": max(1, int(page))}
        data = self._request(
            self.platform_route,
            f"/lol/league/v4/entries/{queue}/{tier}/{division}",
            params,
        )
        return data or []

    def get_high_tier_league(self, queue: str, tier: str) -> List[Dict]:
        path_map = {
            "MASTER": "/lol/league/v4/masterleagues/by-queue/",
            "GRANDMASTER": "/lol/league/v4/grandmasterleagues/by-queue/",
            "CHALLENGER": "/lol/league/v4/challengerleagues/by-queue/",
        }
        path = path_map.get(tier.upper())
        if not path:
            return []
        data = self._request(self.platform_route, f"{path}{queue}")
        if not data:
            return []
        return data.get("entries", []) if isinstance(data, dict) else data

    def get_summoner_by_id(self, summoner_id: str) -> Optional[Dict]:
        return self._request(
            self.platform_route, f"/lol/summoner/v4/summoners/{summoner_id}"
        )

    # Resolve a Riot ID (gameName#tagLine) to a PUUID using account-v1 across clusters
    def get_puuid_by_riot_id(self, game_name: str, tag_line: str) -> Optional[str]:
        account_clusters = [
            "https://europe.api.riotgames.com",
            "https://americas.api.riotgames.com",
            "https://asia.api.riotgames.com",
        ]
        for cluster in account_clusters:
            try:
                data = self._request(
                    cluster,
                    f"/riot/account/v1/accounts/by-riot-id/{requests.utils.quote(game_name)}/{requests.utils.quote(tag_line)}",
                )
                if data and data.get("puuid"):
                    return data["puuid"]
            except requests.HTTPError as e:
                if self.debug:
                    print(f"[debug] account lookup failed on {cluster} for {game_name}#{tag_line}: {e}")
                continue
        return None


class ChampionBalanceAnalyzer:
    def __init__(
        self,
        client: RiotAPIClient,
        config: ChampionBalanceConfig,
        player_puuids: Iterable[str],
    ) -> None:
        self.client = client
        self.config = config
        self.player_puuids = list(player_puuids)
        if not self.player_puuids:
            raise ValueError("At least one player PUUID is required for the search.")
        self.target_name_canonical = _normalize_champion_name(config.champion_name)
        self.rank_cache: Dict[str, Dict] = {}
        self.processed_matches: set[str] = set()
        self.version_counter: Counter = Counter()

    def collect(self) -> Dict[str, Dict]:
        patch_accumulators = {
            alias: PatchAccumulator(
                patch=patch,
                target_matches=self.config.matches_per_patch,
                store_match_ids=self.config.store_match_ids,
            )
            for alias, patch in self.config.patches.items()
        }

        start_time = time.time()
        timeout = self.config.match_search_timeout_seconds
        per_player_offsets = {puuid: 0 for puuid in self.player_puuids}
        per_player_finished = {puuid: False for puuid in self.player_puuids}

        while (
            not self._requirements_met(patch_accumulators)
            and (time.time() - start_time) < timeout
        ):
            progress_made = False

            for puuid in self.player_puuids:
                if self._requirements_met(patch_accumulators):
                    break
                if per_player_finished.get(puuid):
                    continue
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break

                current_offset = per_player_offsets[puuid]
                if current_offset >= self.config.max_matches_per_player:
                    per_player_finished[puuid] = True
                    continue

                remaining_for_player = self.config.max_matches_per_player - current_offset
                batch_size = min(self.config.matches_batch_size, remaining_for_player)
                if batch_size <= 0:
                    per_player_finished[puuid] = True
                    continue

                try:
                    match_ids = self.client.get_match_ids(
                        puuid, start=current_offset, count=batch_size, match_type=self.config.match_type
                    )
                except Exception as exc:
                    print(f"[warn] Failed to get matches for puuid {puuid}: {exc}")
                    # If we can't fetch further for this player, mark finished
                    per_player_finished[puuid] = True
                    continue

                if not match_ids:
                    per_player_finished[puuid] = True
                    continue

                per_player_offsets[puuid] += len(match_ids)
                if len(match_ids) < batch_size:
                    per_player_finished[puuid] = True

                for match_id in match_ids:
                    if self._requirements_met(patch_accumulators):
                        break
                    if match_id in self.processed_matches:
                        continue
                    try:
                        match_data = self.client.get_match(match_id)
                    except Exception as exc:
                        print(f"[warn] Failed to fetch match {match_id}: {exc}")
                        continue
                    if not match_data:
                        continue
                    processed = self._ingest_match(match_data, patch_accumulators)
                    if processed:
                        progress_made = True
                    self.processed_matches.add(match_id)

            if not progress_made and all(per_player_finished.values()):
                break
            if not progress_made:
                time.sleep(1.0)

        elapsed = time.time() - start_time
        timeout_reached = not self._requirements_met(patch_accumulators)

        summaries: Dict[str, Dict] = {}
        for alias, accumulator in patch_accumulators.items():
            summary = accumulator.summarize()
            summary["alias"] = alias
            summaries[alias] = summary

        metadata = {
            "elapsed_seconds": elapsed,
            "timeout_reached": timeout_reached,
            "unique_matches_processed": len(self.processed_matches),
            "players_considered": len(self.player_puuids),
            "target_champion": self.config.champion_name,
            "versions_encountered": dict(self.version_counter),
        }

        return {"metadata": metadata, "patches": summaries}

    @staticmethod
    def _requirements_met(accumulators: Dict[str, PatchAccumulator]) -> bool:
        return all(
            accumulator.champion_matches >= accumulator.target_matches
            for accumulator in accumulators.values()
        )

    def _ingest_match(
        self, match_data: Dict, accumulators: Dict[str, PatchAccumulator]
    ) -> bool:
        info = match_data.get("info", {})
        metadata = match_data.get("metadata", {})
        game_version = info.get("gameVersion")
        if game_version:
            self.version_counter[_short_patch(game_version)] += 1

        patch_alias = self._resolve_patch(game_version)
        if patch_alias is None:
            return False

        queue_id = info.get("queueId")
        if queue_id not in self.config.queues:
            return False

        accumulator = accumulators[patch_alias]
        target_banned = self._is_target_banned(info.get("teams"))
        accumulator.record_match(queue_id, target_banned, game_version)

        participant = self._find_target_participant(info.get("participants"))
        match_id = metadata.get("matchId")

        if participant:
            rank_info = self._get_rank_info(participant)
            accumulator.record_champion(match_id, participant, rank_info)
            return True

        return target_banned

    def _resolve_patch(self, game_version: Optional[str]) -> Optional[str]:
        if not game_version:
            return None
        prefix = ".".join(game_version.split(".")[:2])
        for alias, patch in self.config.patches.items():
            if prefix == patch:
                return alias
        return None

    def _is_target_banned(self, teams: Optional[List[Dict]]) -> bool:
        if not teams:
            return False
        for team in teams:
            for ban in team.get("bans", []):
                champion_id = ban.get("championId")
                if champion_id is None or champion_id == -1:
                    continue
                name = id_to_name(champion_id)
                if name and _normalize_champion_name(name) == self.target_name_canonical:
                    return True
        return False

    def _find_target_participant(
        self, participants: Optional[List[Dict]]
    ) -> Optional[Dict]:
        if not participants:
            return None
        for participant in participants:
            champion_name = participant.get("championName")
            if isinstance(champion_name, str):
                if (
                    _normalize_champion_name(champion_name)
                    == self.target_name_canonical
                ):
                    return participant
            else:
                champion_id = participant.get("championId")
                if champion_id is None:
                    continue
                name = id_to_name(champion_id)
                if name and _normalize_champion_name(name) == self.target_name_canonical:
                    return participant
        return None

    def _get_rank_info(self, participant: Dict) -> Dict:
        # Rank lookup can fail across platforms; treat failures as UNRANKED.
        summoner_id = participant.get("summonerId")
        if not summoner_id:
            return {"tier": "UNRANKED", "rank": "", "leaguePoints": 0}
        if summoner_id in self.rank_cache:
            return self.rank_cache[summoner_id]
        try:
            entries = self.client.get_league_entries(summoner_id)
        except Exception as exc:
            print(f"[warn] Failed to get league entries for summoner: {exc}")
            entries = []
        rank_info = self._select_rank_entry(entries)
        self.rank_cache[summoner_id] = rank_info
        return rank_info

    @staticmethod
    def _select_rank_entry(entries: List[Dict]) -> Dict:
        if not entries:
            return {"tier": "UNRANKED", "rank": "", "leaguePoints": 0}
        solo = next(
            (entry for entry in entries if entry.get("queueType") == "RANKED_SOLO_5x5"),
            None,
        )
        flex = next(
            (entry for entry in entries if entry.get("queueType") == "RANKED_FLEX_SR"),
            None,
        )
        entry = solo or flex or entries[0]
        return {
            "tier": entry.get("tier", "UNRANKED"),
            "rank": entry.get("rank", ""),
            "leaguePoints": entry.get("leaguePoints", 0),
            "queueType": entry.get("queueType"),
        }


def build_config_from_env() -> ChampionBalanceConfig:
    champion_name = os.getenv("TARGET_CHAMPION", "Volibear")
    patches = _infer_default_patches()
    current_patch_env = os.getenv("CURRENT_PATCH")
    if current_patch_env:
        patches["current"] = current_patch_env.strip()
    previous_patch_env = os.getenv("PREVIOUS_PATCH")
    if previous_patch_env:
        patches["previous"] = previous_patch_env.strip()
    try:
        matches_per_patch = int(os.getenv("MATCHES_PER_PATCH", "40"))
    except ValueError:
        matches_per_patch = 40
    try:
        match_search_timeout = int(os.getenv("MATCH_SEARCH_TIMEOUT_SECONDS", "600"))
    except ValueError:
        match_search_timeout = 600
    try:
        max_matches_per_player = int(os.getenv("MAX_MATCHES_PER_PLAYER", "200"))
    except ValueError:
        max_matches_per_player = 200
    try:
        matches_batch_size = int(os.getenv("MATCHES_BATCH_SIZE", "50"))
    except ValueError:
        matches_batch_size = 50
    store_match_ids_env = os.getenv("STORE_MATCH_IDS", "1").lower()
    store_match_ids = store_match_ids_env not in {"0", "false", "no"}
    match_type = os.getenv("MATCH_TYPE", "ranked").lower().strip()
    debug_requests = os.getenv("DEBUG_RIOT_API", "0").lower() in {"1", "true", "yes"}

    config_kwargs = {
        "champion_name": champion_name,
        "patches": patches,
        "matches_per_patch": matches_per_patch,
        "match_search_timeout_seconds": match_search_timeout,
        "queues": _parse_queue_ids(os.getenv("RANKED_QUEUES")),
        "max_matches_per_player": max_matches_per_player,
        "matches_batch_size": matches_batch_size,
        "output_path": os.getenv("BALANCE_OUTPUT_PATH", "champion_balance_stats.json"),
        "store_match_ids": store_match_ids,
        "match_type": match_type,
        "debug_requests": debug_requests,
    }

    seed_strategy = os.getenv("PLAYER_SEED_STRATEGY")
    if seed_strategy:
        config_kwargs["seed_strategy"] = seed_strategy
    seed_queue = os.getenv("SEED_LEAGUE_QUEUE")
    if seed_queue:
        config_kwargs["seed_league_queue"] = seed_queue
    seed_tiers = _parse_csv(os.getenv("SEED_LEAGUE_TIERS"))
    if seed_tiers:
        config_kwargs["seed_league_tiers"] = seed_tiers
    seed_divisions = _parse_csv(os.getenv("SEED_LEAGUE_DIVISIONS"))
    if seed_divisions:
        config_kwargs["seed_league_divisions"] = seed_divisions
    try:
        entries_per_tier = int(os.getenv("SEED_ENTRIES_PER_TIER", "").strip() or "0")
    except ValueError:
        entries_per_tier = 0
    if entries_per_tier > 0:
        config_kwargs["seed_entries_per_tier"] = entries_per_tier

    player_puuids_override = _parse_csv(os.getenv("PLAYER_PUUIDS"))
    if player_puuids_override:
        config_kwargs["player_puuids_override"] = player_puuids_override
    player_puuid_file = os.getenv("PLAYER_PUUIDS_FILE")
    if player_puuid_file:
        config_kwargs["player_puuid_file"] = player_puuid_file
    summary_path = os.getenv("BALANCE_SUMMARY_PATH")
    if summary_path:
        config_kwargs["summary_output_path"] = summary_path
    plot_toggle = os.getenv("BALANCE_PLOT")
    if plot_toggle is not None:
        config_kwargs["generate_plot"] = plot_toggle.strip().lower() not in {"0", "false", "no"}
    plot_path = os.getenv("BALANCE_PLOT_PATH")
    if plot_path:
        config_kwargs["plot_output_path"] = plot_path

    return ChampionBalanceConfig(**config_kwargs)


def _parse_summoner_names_env() -> List[Tuple[str, str]]:
    raw = os.getenv("SUMMONER_NAMES", "").strip()
    if not raw:
        return []
    pairs: List[Tuple[str, str]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "#" in token:
            name, tag = token.split("#", 1)
            pairs.append((name.strip(), tag.strip()))
        else:
            # If no tagline provided, default to EUW
            pairs.append((token, "EUW"))
    return pairs


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _load_puuids_from_file(path: Path) -> List[str]:
    if not path.exists():
        print(f"[warn] Player PUUID file not found: {path}")
        return []
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"[warn] Could not read player PUUID file {path}: {exc}")
        return []
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    return []


def _collect_league_seed_puuids(
    client: RiotAPIClient, config: ChampionBalanceConfig
) -> List[str]:
    puuids: List[str] = []
    for tier in config.seed_league_tiers:
        tier_upper = tier.upper()
        entries: List[Dict] = []
        try:
            if tier_upper in {"MASTER", "GRANDMASTER", "CHALLENGER"}:
                entries = client.get_high_tier_league(config.seed_league_queue, tier_upper)
            else:
                for division in config.seed_league_divisions:
                    if len(entries) >= config.seed_entries_per_tier:
                        break
                    page = 1
                    while len(entries) < config.seed_entries_per_tier:
                        page_entries = client.get_league_entries_page(
                            config.seed_league_queue, tier_upper, division.upper(), page
                        )
                        if not page_entries:
                            break
                        entries.extend(page_entries)
                        if len(page_entries) < 205:  # no more pages
                            break
                        page += 1
            if not entries:
                print(f"[warn] No league entries found for tier {tier_upper}")
                continue
            trimmed = entries[: config.seed_entries_per_tier]
            for entry in trimmed:
                puuid = entry.get("puuid")
                if puuid:
                    puuids.append(puuid)
                    continue
                summoner_id = entry.get("summonerId")
                if not summoner_id:
                    continue
                try:
                    profile = client.get_summoner_by_id(summoner_id)
                except Exception as exc:
                    print(f"[warn] Failed to fetch summoner profile for tier {tier_upper}: {exc}")
                    continue
                if profile and profile.get("puuid"):
                    puuids.append(profile["puuid"])
        except Exception as exc:
            print(f"[warn] Failed to collect seed players for tier {tier_upper}: {exc}")
            continue
    return _dedupe_preserve(puuids)


def _resolve_player_puuids(
    client: RiotAPIClient, config: ChampionBalanceConfig
) -> List[str]:
    # Prefer explicit Riot IDs from env over static list
    pairs = _parse_summoner_names_env()
    puuids: List[str] = list(config.player_puuids_override)
    if config.player_puuid_file:
        puuids.extend(_load_puuids_from_file(Path(config.player_puuid_file)))
    if pairs:
        for name, tag in pairs:
            puuid = client.get_puuid_by_riot_id(name, tag)
            if puuid:
                puuids.append(puuid)
            else:
                print(f"[warn] Could not resolve PUUID for {name}#{tag}")
    puuids = _dedupe_preserve(puuids)

    if puuids:
        return puuids

    # Fallback strategies
    if config.seed_strategy in {"static"}:
        return _dedupe_preserve(summoner_list)

    if config.seed_strategy in {"auto", "league"}:
        league_puuids = _collect_league_seed_puuids(client, config)
        if league_puuids:
            return league_puuids

    # As a last resort, use bundled list.
    if config.seed_strategy == "auto":
        fallback = _dedupe_preserve(summoner_list)
        if fallback:
            print("[warn] Using bundled summoner_list fallback; consider providing SUMMONER_NAMES or PLAYER_PUUIDS.")
            return fallback

    raise RuntimeError(
        "Could not resolve any player PUUIDs. Set SUMMONER_NAMES, PLAYER_PUUIDS, or provide a seed file."
    )


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _write_summary_text(payload: Dict, path: Path) -> None:
    metadata = payload.get("metadata", {})
    summaries = payload.get("patch_summaries", {})
    lines = []
    lines.append(f"Champion: {payload.get('champion', 'n/a')}")
    versions = metadata.get("versions_encountered") or {}
    if versions:
        versions_line = ", ".join(f"{ver}:{count}" for ver, count in versions.items())
        lines.append(f"Versions encountered: {versions_line}")
    lines.append(
        f"Matches processed: {metadata.get('unique_matches_processed', 0)} "
        f"(timeout={'yes' if metadata.get('timeout_reached') else 'no'})"
    )
    lines.append("")
    for alias, summary in summaries.items():
        lines.append(f"[{alias}] Patch {summary.get('patch', 'n/a')}")
        lines.append(
            f"  Champion matches: {summary.get('champion_matches', 0)}/"
            f"{summary.get('target_matches', 0)}"
        )
        lines.append(
            f"  Win rate: {_format_rate(summary.get('win_rate'))} | "
            f"Pick rate: {_format_rate(summary.get('pick_rate'))} | "
            f"Ban rate: {_format_rate(summary.get('ban_rate'))}"
        )
        avg_stats = summary.get("average_stats_per_game") or {}
        if avg_stats:
            lines.append(
                "  Avg stats: "
                f"K/D/A {avg_stats.get('kills', 0):.1f}/"
                f"{avg_stats.get('deaths', 0):.1f}/"
                f"{avg_stats.get('assists', 0):.1f}, "
                f"DMG {avg_stats.get('damage_dealt', 0):.0f}, "
                f"Taken {avg_stats.get('damage_taken', 0):.0f}, "
                f"Gold {avg_stats.get('gold', 0):.0f}, "
                f"Vision {avg_stats.get('vision_score', 0):.1f}"
            )
        observed_versions = summary.get("observed_game_versions") or {}
        if observed_versions:
            joined_versions = ", ".join(
                f"{ver}:{count}" for ver, count in observed_versions.items()
            )
            lines.append(f"  Patch breakdown: {joined_versions}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _save_balance_plot(payload: Dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[warn] matplotlib not available; skipping plot output.")
        return

    summaries = payload.get("patch_summaries", {})
    if not summaries:
        print("[warn] No patch summaries to plot.")
        return

    labels: List[str] = []
    pick_rates: List[float] = []
    win_rates: List[float] = []
    for alias, summary in summaries.items():
        labels.append(f"{alias} ({summary.get('patch', 'n/a')})")
        pick_rates.append(
            (summary.get("pick_rate") or 0.0) * 100.0
        )
        win_rates.append(
            (summary.get("win_rate") or 0.0) * 100.0
        )

    if not any(pick_rates) and not any(win_rates):
        print("[warn] Not enough data to draw plot.")
        return

    x_positions = range(len(labels))
    width = 0.35
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(
        [x - width / 2 for x in x_positions],
        pick_rates,
        width=width,
        label="Pick rate (%)",
        color="#4C78A8",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        win_rates,
        width=width,
        label="Win rate (%)",
        color="#F58518",
    )
    plt.xticks(list(x_positions), labels, rotation=20, ha="right")
    plt.ylabel("Percentage (%)")
    plt.title(f"Champion balance snapshot: {payload.get('champion', '')}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved balance plot to {path}")


def main() -> None:
    api_key = os.getenv("LEAGUE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LEAGUE_API_KEY is not set. Please add it to your .env file or environment."
        )

    config = build_config_from_env()
    print(
        f"[*] Starting balance snapshot for {config.champion_name} "
        f"across patches {list(config.patches.values())} (type={config.match_type})"
    )

    client = RiotAPIClient(api_key, debug=config.debug_requests)
    player_puuids = _resolve_player_puuids(client, config)
    analyzer = ChampionBalanceAnalyzer(client, config, player_puuids)
    result = analyzer.collect()

    output_payload = {
        "champion": config.champion_name,
        "config": asdict(config),
        "metadata": result["metadata"],
        "patch_summaries": result["patches"],
    }

    with open(config.output_path, "w", encoding="utf-8") as output_file:
        json.dump(output_payload, output_file, indent=2)

    summary_path = Path(config.summary_output_path)
    _write_summary_text(output_payload, summary_path)
    print(f"[+] Saved balance summary to {summary_path}")

    if config.generate_plot:
        if config.plot_output_path:
            plot_path = Path(config.plot_output_path)
        else:
            base = Path(config.output_path)
            plot_path = base.with_name(f"{base.stem}_plot.png")
        _save_balance_plot(output_payload, plot_path)

    print(f"[+] Saved champion balance snapshot to {config.output_path}")
    for alias, summary in result["patches"].items():
        completed = summary["champion_matches"] >= summary["target_matches"]
        status_icon = "[done]" if completed else "[partial]"
        pick_rate = summary["pick_rate"]
        win_rate = summary["win_rate"]
        pick_rate_label = f"{pick_rate:.1%}" if pick_rate is not None else "n/a"
        win_rate_label = f"{win_rate:.1%}" if win_rate is not None else "n/a"
        print(
            f"{status_icon} {alias} ({summary['patch']}): "
            f"{summary['champion_matches']}/{summary['target_matches']} matches "
            f"| win rate {win_rate_label} | pick rate {pick_rate_label} "
            f"| bans {summary['ban_count']}"
        )

    if result["metadata"]["timeout_reached"]:
        print(
            "[warn] Timeout reached before gathering all requested matches. "
            "Consider increasing MATCH_SEARCH_TIMEOUT_SECONDS or MAX_MATCHES_PER_PLAYER."
        )


if __name__ == "__main__":
    main()
