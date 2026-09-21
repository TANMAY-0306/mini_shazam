from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import psycopg2
from db.database import get_connection
import config

# Matching thresholds
MIN_MATCH_THRESHOLD: int = 15  # Minimum landmark consensus matches required to confirm identity


def query_matching_fingerprints(hashes: List[str]) -> List[Tuple[int, int, str]]:
    """
    Queries Neon DB for all fingerprints that match the query hash set.
    Returns: List of (song_id, offset_frame, hash).
    """
    if not hashes:
        return []

    # Filter to unique hashes to avoid unnecessary SQL payload size
    unique_hashes = list(set(hashes))

    query = """
        SELECT song_id, offset_frame, hash
        FROM fingerprints
        WHERE hash = ANY(%s);
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (unique_hashes,))
            results = cur.fetchall()
            return results


def find_best_match(
    query_hashes: List[Tuple[str, int]],
    min_threshold: int = MIN_MATCH_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    Runs Avery Wang's histogram consensus algorithm to match query hashes against the database.
    
    Args:
        query_hashes: List of (hash, t_query_frame) extracted from query/mic audio.
        min_threshold: Minimum peak consensus count to accept a match.

    Returns:
        Dict with matched song details or None if no confident match is found.
    """
    if not query_hashes:
        return None

    # Map query hashes: hash -> list of query timestamps (in frames)
    query_hash_map = defaultdict(list)
    for h, t_q in query_hashes:
        query_hash_map[h].append(t_q)

    # 1. Fetch matching candidate records from DB
    raw_hashes = list(query_hash_map.keys())
    db_matches = query_matching_fingerprints(raw_hashes)

    if not db_matches:
        return None

    # 2. Compute time-offset differences: delta_offset = t_db - t_query
    # Structure: song_offsets[song_id][delta_offset] = count
    song_offsets = defaultdict(lambda: defaultdict(int))

    for song_id, t_db, h in db_matches:
        for t_query in query_hash_map[h]:
            delta_offset = t_db - t_query
            song_offsets[song_id][delta_offset] += 1

    # 3. Locate the highest peak across all candidate songs
    best_song_id = None
    best_consensus_count = 0
    best_delta_offset = 0

    for song_id, offsets in song_offsets.items():
        # Find the single offset with the highest alignment for this song
        top_offset = max(offsets, key=offsets.get)
        count = offsets[top_offset]

        if count > best_consensus_count:
            best_consensus_count = count
            best_song_id = song_id
            best_delta_offset = top_offset

    if best_consensus_count < min_threshold or best_song_id is None:
        return None

    # 4. Fetch song metadata from DB
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, artist, duration_sec FROM songs WHERE id = %s;",
                (best_song_id,)
            )
            song = cur.fetchone()

    if not song:
        return None

    # Convert offset from frames to seconds
    offset_seconds = best_delta_offset * (config.HOP_LENGTH / config.TARGET_SR)

    return {
        "song_id": song[0],
        "title": song[1],
        "artist": song[2],
        "duration_sec": song[3],
        "confidence_score": best_consensus_count,
        "offset_seconds": max(0.0, offset_seconds)
    }