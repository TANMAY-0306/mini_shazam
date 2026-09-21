import sys
from audio.loader import load_audio
from audio.preprocessor import preprocess_audio
from audio.spectrogram import compute_spectrogram
from fingerprint.peaks import extract_peaks, PEAK_NEIGHBORHOOD_SIZE, MIN_AMPLITUDE_DB
from fingerprint.hasher import generate_hashes, FAN_OUT, MIN_DELTA_TIME, MAX_DELTA_TIME
import config


def test_hashing_pipeline(file_path: str):
    print(f"\n--- Testing Fingerprint Hashing for: {file_path} ---")

    # 1. Full pipeline through peaks
    raw_audio, sr = load_audio(file_path)
    mono_audio, final_sr = preprocess_audio(raw_audio, orig_sr=sr, target_sr=config.TARGET_SR)
    spec_db = compute_spectrogram(mono_audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    peaks = extract_peaks(spec_db, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, min_amplitude_db=MIN_AMPLITUDE_DB)

    # 2. Generate hashes
    fingerprints = generate_hashes(
        peaks,
        fan_out=FAN_OUT,
        min_delta_time=MIN_DELTA_TIME,
        max_delta_time=MAX_DELTA_TIME
    )

    duration_sec = len(mono_audio) / final_sr
    hashes_per_sec = len(fingerprints) / duration_sec if duration_sec > 0 else 0

    # 3. Assertions
    assert len(fingerprints) > 0, "No hashes were generated!"
    assert all(len(h[0]) == 20 for h in fingerprints), "Hash digest length mismatch!"
    

    print("Hashing Pipeline Successful:")
    print(f"  • Input Peaks:         {len(peaks):,}")
    print(f"  • Generated Hashes:    {len(fingerprints):,}")
    print(f"  • Hash Density:        {hashes_per_sec:.2f} hashes/second")
    print(f"  • Fan-out Factor:      {FAN_OUT}")
    print(f"  • Delta Time Bounds:   [{MIN_DELTA_TIME}, {MAX_DELTA_TIME}] frames")
    print("\nSample Generated Fingerprints (hash, t1_time_frame):")
    for hash_token, t1_time_frame in fingerprints[:5]:
        time_sec = t1_time_frame * (config.HOP_LENGTH / config.TARGET_SR)
        print(f"    hash: {hash_token}  -->  t1_time_frame: {t1_time_frame} ({time_sec:.3f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_hasher path/to/song.mp3")
    else:
        test_hashing_pipeline(sys.argv[1])