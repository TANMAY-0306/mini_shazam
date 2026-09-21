import sys
import hashlib
from pathlib import Path
from audio.loader import load_audio
from audio.preprocessor import preprocess_audio
from audio.spectrogram import compute_spectrogram
from fingerprint.peaks import extract_peaks, PEAK_NEIGHBORHOOD_SIZE, MIN_AMPLITUDE_DB
from fingerprint.hasher import generate_hashes, FAN_OUT, MIN_DELTA_TIME, MAX_DELTA_TIME
from db.database import song_exists, insert_song, store_fingerprints
import config


def compute_file_hash(filepath: Path) -> str:
    """Computes a SHA-256 hash of the raw audio file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def index_audio_file(filepath: str, title: str | None = None, artist: str | None = None):
    path = Path(filepath).resolve()
    if not path.is_file():
        print(f"Error: File '{path}' does not exist.")
        return

    # Derive track title and artist from filename if not explicitly provided
    # Format expectation: "Artist - Title.mp3" or fallback to filename stem
    if not title or not artist:
        parts = path.stem.split(" - ")
        if len(parts) >= 2:
            artist = artist or parts[0].strip()
            title = title or " - ".join(parts[1:]).strip()
        else:
            artist = artist or "Unknown"
            title = title or path.stem

    print(f"\n--- Indexing Audio File: '{title}' by '{artist}' ---")
    
    # 1. Compute file content hash for deduplication
    file_hash = compute_file_hash(path)
    existing_id = song_exists(file_hash)
    if existing_id:
        print(f"Song already indexed in database (song_id = {existing_id}). Skipping insertion.")
        return existing_id

    # 2. Ingest and preprocess
    print("1/4 Loading audio waveform...")
    raw_audio, native_sr = load_audio(path)
    
    print("2/4 Preprocessing (mono downmix, resample to 44.1kHz, peak normalize)...")
    mono_audio, final_sr = preprocess_audio(raw_audio, orig_sr=native_sr, target_sr=config.TARGET_SR)
    duration_sec = len(mono_audio) / final_sr

    # 3. Spectrogram and peak extraction
    print("3/4 Generating STFT spectrogram and extracting constellation peaks...")
    spec_db = compute_spectrogram(mono_audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    peaks = extract_peaks(spec_db, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, min_amplitude_db=MIN_AMPLITUDE_DB)
    print(f"    Extracted {len(peaks):,} peaks ({len(peaks) / duration_sec:.2f} peaks/s)")

    # 4. Generate combinatorial landmark hashes
    print("4/4 Generating fingerprint hashes...")
    hashes = generate_hashes(
        peaks,
        fan_out=FAN_OUT,
        min_delta_time=MIN_DELTA_TIME,
        max_delta_time=MAX_DELTA_TIME
    )
    print(f"    Generated {len(hashes):,} hashes")

    # 5. Store in Neon DB
    print("Storing records in Neon PostgreSQL...")
    song_id = insert_song(title=title, artist=artist, file_hash=file_hash, duration_sec=duration_sec)
    store_fingerprints(song_id=song_id, hashes=hashes, page_size=10000)

    print(f"\nIndexing Complete:")
    print(f"  • Song ID:          {song_id}")
    print(f"  • Title:            {title}")
    print(f"  • Artist:           {artist}")
    print(f"  • Duration:         {duration_sec:.2f} s")
    print(f"  • Hashes Stored:    {len(hashes):,}")
    return song_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.index_song \"path/to/song.mp3\" [title] [artist]")
    else:
        song_path = sys.argv[1]
        song_title = sys.argv[2] if len(sys.argv) > 2 else None
        song_artist = sys.argv[3] if len(sys.argv) > 3 else None
        index_audio_file(song_path, song_title, song_artist)