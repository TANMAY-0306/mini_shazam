-- Table 1: Metadata for registered audio tracks
CREATE TABLE IF NOT EXISTS songs (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist VARCHAR(255) DEFAULT 'Unknown',
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    duration_sec REAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Landmark fingerprint tokens
CREATE TABLE IF NOT EXISTS fingerprints (
    id BIGSERIAL PRIMARY KEY,
    hash CHAR(20) NOT NULL,
    song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    offset_frame INTEGER NOT NULL
);

-- Index for fast hash lookup
CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints(hash);

-- Composite index to accelerate song-specific lookups or deletions
CREATE INDEX IF NOT EXISTS idx_fingerprints_song_hash ON fingerprints(song_id, hash);