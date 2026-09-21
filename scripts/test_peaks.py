import sys
import numpy as np
from audio.loader import load_audio
from audio.preprocessor import preprocess_audio
from audio.spectrogram import compute_spectrogram
from fingerprint.peaks import extract_peaks, PEAK_NEIGHBORHOOD_SIZE, MIN_AMPLITUDE_DB
import config


def test_peaks_pipeline(file_path: str):
    print(f"\n--- Testing Constellation Peak Extraction for: {file_path} ---")

    # 1. Pipeline: Load -> Preprocess -> Spectrogram
    raw_audio, sr = load_audio(file_path)
    mono_audio, final_sr = preprocess_audio(raw_audio, orig_sr=sr, target_sr=config.TARGET_SR)
    spec_db = compute_spectrogram(mono_audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)

    # 2. Extract peaks
    peaks = extract_peaks(
        spec_db,
        neighborhood_size=PEAK_NEIGHBORHOOD_SIZE,
        min_amplitude_db=MIN_AMPLITUDE_DB
    )

    duration_sec = len(mono_audio) / final_sr
    peaks_per_sec = len(peaks) / duration_sec if duration_sec > 0 else 0

    # 3. Verification checks
    assert len(peaks) > 0, "Error: No peaks extracted! Check threshold or audio input."
    assert all(0 <= f < spec_db.shape[0] and 0 <= t < spec_db.shape[1] for f, t in peaks), "Peak coordinates out of bounds!"

    print("Peaks Extracted Successfully:")
    print(f"  • Total Peaks:            {len(peaks):,}")
    print(f"  • Audio Duration:         {duration_sec:.2f} seconds")
    print(f"  • Peak Density:           {peaks_per_sec:.2f} peaks/second")
    print(f"  • Neighborhood Footprint: {PEAK_NEIGHBORHOOD_SIZE} (Freq x Time)")
    print(f"  • Min Amplitude Filter:   {MIN_AMPLITUDE_DB} dB")
    print(f"  • Sample First 5 Peaks:   {peaks[:5]} (freq_bin, time_frame)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_peaks path/to/song.mp3")
    else:
        test_peaks_pipeline(sys.argv[1])