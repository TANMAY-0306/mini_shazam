from pathlib import Path
from typing import List, Tuple, Optional
import psycopg2
from psycopg2.extras import execute_values
import config


def get_connection():
    """Establishes an authenticated SSL connection to the Neon DB."""
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        sslmode=config.DB_SSLMODE,
        connect_timeout=10
    )


def init_db(schema_path: str = "db/schema.sql"):
    """Reads schema.sql and creates tables and indexes."""
    path = Path(schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"Schema file not found at: {path.resolve()}")

    with open(path, "r") as f:
        schema_sql = f.read()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()


def song_exists(file_hash: str) -> Optional[int]:
    """Checks if a song with the content hash already exists."""
    query = "SELECT id FROM songs WHERE file_hash = %s;"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (file_hash,))
            row = cur.fetchone()
            return row[0] if row else None


def insert_song(title: str, artist: str, file_hash: str, duration_sec: float) -> int:
    """Inserts track metadata and returns the generated song_id."""
    query = """
        INSERT INTO songs (title, artist, file_hash, duration_sec)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (title, artist, file_hash, duration_sec))
            song_id = cur.fetchone()[0]
        conn.commit()
    return song_id


def store_fingerprints(song_id: int, hashes: List[Tuple[str, int]], page_size: int = 10000):
    """
    Bulk-inserts fingerprints using psycopg2.extras.execute_values.
    
    Args:
        song_id: Foreign key ID of the parent song.
        hashes: List of (hash, t1_time_frame) tuples.
        page_size: Batch size for bulk insertion.
    """
    if not hashes:
        return

    records = [(hash_val, song_id, t1_time_frame) for hash_val, t1_time_frame in hashes]

    query = """
        INSERT INTO fingerprints (hash, song_id, offset_frame)
        VALUES %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, records, template="(%s, %s, %s)", page_size=page_size)
        conn.commit()