import sys
from pathlib import Path
from audio.loader import load_audio, AudioLoadingError, EmptyAudioError


def test_loader(file_path: str):
    print(f"\n--- Testing Audio Ingestion for: {file_path} ---")
    try:
        waveform, sr = load_audio(file_path)
        duration = waveform.shape[-1] / sr
        channels = 1 if waveform.ndim == 1 else waveform.shape[0]

        print(f"Loaded successfully!")
        print(f"  • Shape:        {waveform.shape}")
        print(f"  • Data type:    {waveform.dtype}")
        print(f"  • Sample Rate:  {sr} Hz")
        print(f"  • Channels:     {channels} ({'Mono' if channels == 1 else 'Stereo'})")
        print(f"  • Duration:     {duration:.2f} seconds")
        print(f"  • Amplitude Range: [{waveform.min():.4f}, {waveform.max():.4f}]")
    except Exception as e:
        print(f"Error loading audio: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_load path/to/your/audio_file.mp3")
    else:
        test_loader(sys.argv[1])