from pathlib import Path
from typing import Tuple
import numpy as np
import librosa


class AudioLoadingError(Exception):
    """Raised when an audio file cannot be loaded or decoded."""
    pass


class EmptyAudioError(Exception):
    """Raised when the loaded audio contains zero samples or is silent."""
    pass


def load_audio(filepath: str | Path, sr: int | None = None) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file into a floating-point time series.

    Args:
        filepath: Path to the target audio file (mp3, wav, flac, etc.).
        sr: Target sample rate. If None, preserves the file's native sample rate.

    Returns:
        Tuple[np.ndarray, int]: 
            - waveform: 1D or 2D numpy array of audio samples (float32).
            - sample_rate: The sample rate of the returned waveform.

    Raises:
        FileNotFoundError: If filepath does not exist.
        EmptyAudioError: If decoded audio contains no samples.
        AudioLoadingError: If the codec or file format fails to decode.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found at: {path.resolve()}")

    try:
        # sr=None preserves native sampling rate for inspection
        # mono=False preserves channel structure (stereo/mono) for Phase 2 inspection
        waveform, native_sr = librosa.load(str(path), sr=sr, mono=False)
    except Exception as exc:
        raise AudioLoadingError(f"Failed to decode audio file '{path.name}': {exc}") from exc

    if waveform is None or waveform.size == 0:
        raise EmptyAudioError(f"Audio file '{path.name}' is empty or contains no samples.")

    return waveform, native_sr