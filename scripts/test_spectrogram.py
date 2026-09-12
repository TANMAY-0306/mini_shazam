import sys
import numpy as np
from audio.loader import load_audio
from audio.preprocessor import preprocess_audio
from audio.spectrogram import compute_spectrogram
import config


def test_spectrogram_pipeline(file_path: str):
    print(f"\n--- Testing Spectrogram Generation for: {file_path} ---")

    # 1. Load and preprocess
    raw_audio, sr = load_audio(file_path)
    mono_audio, final_sr = preprocess_audio(raw_audio, orig_sr=sr, target_sr=config.TARGET_SR)

    # 2. Compute spectrogram
    spec_db = compute_spectrogram(mono_audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)

    expected_freq_bins = 1 + config.N_FFT // 2  # 1 + 4096 // 2 = 2049
    expected_frames = 1 + len(mono_audio) // config.HOP_LENGTH

    # 3. Verifications
    assert spec_db.ndim == 2, f"Expected 2D matrix, got shape {spec_db.shape}"
    assert spec_db.shape[0] == expected_freq_bins, f"Expected {expected_freq_bins} frequency bins, got {spec_db.shape[0]}"
    assert np.isclose(spec_db.max(), 0.0, atol=1e-2), f"Expected peak at 0 dB, got {spec_db.max()}"

    freq_resolution = final_sr / config.N_FFT
    time_resolution_ms = (config.HOP_LENGTH / final_sr) * 1000

    print("Spectrogram Computed Successfully:")
    print(f"  • Matrix Shape:       {spec_db.shape} (Freq Bins x Time Frames)")
    print(f"  • Frequency Bins:     {spec_db.shape[0]} (Range: 0 Hz to {final_sr / 2:.0f} Hz)")
    print(f"  • Time Frames:        {spec_db.shape[1]} columns")
    print(f"  • Frequency Step:     {freq_resolution:.2f} Hz per bin")
    print(f"  • Time Step:          {time_resolution_ms:.2f} ms per frame")
    print(f"  • Dynamic Range (dB): [{spec_db.min():.2f} dB, {spec_db.max():.2f} dB]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_spectrogram path/to/song.mp3")
    else:
        test_spectrogram_pipeline(sys.argv[1])