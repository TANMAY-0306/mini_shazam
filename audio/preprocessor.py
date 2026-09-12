import numpy as np
import librosa
from config import TARGET_SR


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """
    Downmixes a multi-channel audio array to single-channel mono.

    Args:
        waveform: 1D or 2D numpy array.

    Returns:
        np.ndarray: 1D mono audio array.
    """
    if waveform.ndim == 1:
        return waveform
    elif waveform.ndim == 2:
        return np.mean(waveform, axis=0)
    else:
        raise ValueError(f"Unsupported audio shape {waveform.shape}; expected 1D or 2D array.")


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Resamples a waveform to the target sample rate.

    Args:
        waveform: 1D numpy array of audio samples.
        orig_sr: Original sample rate of input audio.
        target_sr: Desired output sample rate.

    Returns:
        np.ndarray: Resampled 1D audio array.
    """
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(y=waveform, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Peak-normalizes the audio array to the [-1.0, 1.0] range.

    Args:
        waveform: 1D numpy array.
        eps: Small constant to avoid division by zero.

    Returns:
        np.ndarray: Peak-normalized 1D audio array.
    """
    max_peak = np.max(np.abs(waveform))
    if max_peak < eps:
        return waveform
    return waveform / (max_peak + eps)


def preprocess_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline: Stereo-to-Mono -> Resample -> Normalize.

    Args:
        waveform: Raw audio array (1D or 2D).
        orig_sr: Input sample rate.
        target_sr: Standardized target sample rate.

    Returns:
        tuple[np.ndarray, int]: (processed_mono_waveform, target_sr)
    """
    mono = to_mono(waveform)
    resampled = resample_audio(mono, orig_sr=orig_sr, target_sr=target_sr)
    normalized = normalize_audio(resampled)
    return normalized.astype(np.float32), target_sr