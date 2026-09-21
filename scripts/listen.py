import time
import numpy as np
import sounddevice as sd

from audio.preprocessor import preprocess_audio
from audio.spectrogram import compute_spectrogram
from fingerprint.peaks import extract_peaks, PEAK_NEIGHBORHOOD_SIZE, MIN_AMPLITUDE_DB
from fingerprint.hasher import generate_hashes, FAN_OUT, MIN_DELTA_TIME, MAX_DELTA_TIME
from recognition.matcher import find_best_match
import config

RECORD_SECONDS: float = 7.0


def listen_and_recognize(duration: float = RECORD_SECONDS):
    sample_rate = config.TARGET_SR
    total_samples = int(duration * sample_rate)

    print(f"\nListening for {duration:.1f} seconds... (Play any of your 53 songs!)")
    audio_buffer = sd.rec(total_samples, samplerate=sample_rate, channels=1, dtype='float32')

    for remaining in range(int(duration), 0, -1):
        print(f"  Recording... {remaining}s left", end="\r", flush=True)
        time.sleep(1)

    sd.wait()
    print("\nRecording finished! Processing audio...")

    raw_audio = audio_buffer.flatten()

    # Preprocess
    mono_audio, final_sr = preprocess_audio(raw_audio, orig_sr=sample_rate, target_sr=sample_rate)

    # Feature extraction
    spec_db = compute_spectrogram(mono_audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    peaks = extract_peaks(spec_db, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, min_amplitude_db=MIN_AMPLITUDE_DB)
    query_hashes = generate_hashes(peaks, fan_out=FAN_OUT, min_delta_time=MIN_DELTA_TIME, max_delta_time=MAX_DELTA_TIME)

    print(f"Captured {len(peaks):,} peaks and {len(query_hashes):,} hashes.")

    # Match
    match = find_best_match(query_hashes)

    if match:
        mins = int(match['offset_seconds'] // 60)
        secs = int(match['offset_seconds'] % 60)
        print("\n=======================================================")
        print("  SONG IDENTIFIED!")
        print("=======================================================")
        print(f"  Title:            {match['title']}")
        print(f"  Artist:           {match['artist']}")
        print(f"  Confidence Score: {match['confidence_score']} aligned landmark pairs")
        print(f"  Playback Cue:     ~{mins:02d}:{secs:02d}")
        print("=======================================================\n")
    else:
        print("\nNo confident match found. Try playing louder or holding the mic closer.")


if __name__ == "__main__":
    listen_and_recognize()