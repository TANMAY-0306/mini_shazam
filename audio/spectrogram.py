import numpy as np
import librosa
from config import N_FFT, HOP_LENGTH


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Computes the complex Short-Time Fourier Transform (STFT) of a 1D audio signal.

    Args:
        waveform: 1D normalized float32 audio array.
        n_fft: FFT window size in samples.
        hop_length: Hop step size between successive frames in samples.

    Returns:
        np.ndarray: 2D complex STFT matrix of shape (1 + n_fft // 2, num_frames).
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D mono audio array, got shape {waveform.shape}")

    # librosa.stft applies a Hann window by default
    stft_complex = librosa.stft(
        y=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True
    )
    return stft_complex


def compute_spectrogram(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    ref_max: bool = True
) -> np.ndarray:
    """
    Computes the log-magnitude (dB) spectrogram from a 1D mono audio signal.

    Args:
        waveform: 1D normalized float32 audio array.
        n_fft: FFT window size in samples.
        hop_length: Hop length in samples.
        ref_max: If True, references 0 dB to the maximum magnitude in the signal.

    Returns:
        np.ndarray: 2D real-valued spectrogram in decibels (dB), shape (freq_bins, time_frames).
    """
    # 1. Compute complex STFT
    stft_matrix = compute_stft(waveform, n_fft=n_fft, hop_length=hop_length)

    # 2. Extract linear magnitude: |X|
    magnitude = np.abs(stft_matrix)

    # 3. Convert magnitude to decibel scale: 20 * log10(magnitude / ref)
    ref_val = np.max if ref_max else 1.0
    spectrogram_db = librosa.amplitude_to_db(magnitude, ref=ref_val)

    return spectrogram_db