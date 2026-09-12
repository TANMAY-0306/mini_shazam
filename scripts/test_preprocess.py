import sys
import numpy as np
from audio.loader import load_audio
from audio.preprocessor import preprocess_audio
import config


def test_preprocessing(file_path: str):
    print(f"\n--- Testing Full Preprocessing Pipeline for: {file_path} ---")

    # 1. Load raw audio
    raw_waveform, native_sr = load_audio(file_path)
    print(f"Raw Input:        shape={raw_waveform.shape}, sr={native_sr}, range=[{raw_waveform.min():.4f}, {raw_waveform.max():.4f}]")

    # 2. Run preprocessing
    processed_waveform, final_sr = preprocess_audio(raw_waveform, orig_sr=native_sr, target_sr=config.TARGET_SR)

    # 3. Assertions
    assert processed_waveform.ndim == 1, f"Expected 1D array, got ndim={processed_waveform.ndim}"
    assert final_sr == config.TARGET_SR, f"Expected SR={config.TARGET_SR}, got {final_sr}"
    assert np.isclose(np.max(np.abs(processed_waveform)), 1.0, atol=1e-3), "Waveform was not normalized to peak 1.0"
    assert processed_waveform.dtype == np.float32, f"Expected float32, got {processed_waveform.dtype}"

    print("\nPreprocessed Output:")
    print(f"  • Shape:            {processed_waveform.shape} (1D Mono)")
    print(f"  • Sample Rate:      {final_sr} Hz")
    print(f"  • Peak Value:       {np.max(np.abs(processed_waveform)):.4f}")
    print(f"  • Duration:         {processed_waveform.shape[-1] / final_sr:.2f} seconds")
    print("Preprocessing pipeline verified successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_preprocess path/to/song.mp3")
    else:
        test_preprocessing(sys.argv[1])