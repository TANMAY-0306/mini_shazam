from typing import List, Tuple
import numpy as np
from scipy.ndimage import maximum_filter


# # Default peak detection hyperparameters
# PEAK_NEIGHBORHOOD_SIZE: Tuple[int, int] = (20, 20)  # (frequency_bins, time_frames)
# MIN_AMPLITUDE_DB: float = -35.0                    # Reject peaks quieter than -35 dB
# Updated peak detection hyperparameters
PEAK_NEIGHBORHOOD_SIZE: Tuple[int, int] = (30, 30)  # Expanded window reduces redundant micro-peaks
MIN_AMPLITUDE_DB: float = -30.0                    # Tightened from -35.0 dB to filter quieter harmonics


def extract_peaks(
    spectrogram_db: np.ndarray,
    neighborhood_size: Tuple[int, int] = PEAK_NEIGHBORHOOD_SIZE,
    min_amplitude_db: float = MIN_AMPLITUDE_DB
) -> List[Tuple[int, int]]:
    """
    Extracts prominent 2D local maxima (peaks) from a dB spectrogram.

    Args:
        spectrogram_db: 2D array of spectrogram values in dB, shape (freq_bins, time_frames).
        neighborhood_size: Tuple (freq_span, time_span) defining the local window size.
        min_amplitude_db: Minimum amplitude threshold in dB for a peak to be kept.

    Returns:
        List[Tuple[int, int]]: List of (frequency_bin, time_frame) coordinate pairs.
    """
    if spectrogram_db.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram matrix, got shape {spectrogram_db.shape}")

    # 1. Apply 2D maximum filter across the spectrogram
    local_max = maximum_filter(spectrogram_db, size=neighborhood_size, mode='constant', cval=-np.inf)

    # 2. Identify coordinates where the original value matches the local maximum
    #    and exceeds the minimum decibel threshold
    is_peak = (spectrogram_db == local_max) & (spectrogram_db >= min_amplitude_db)

    # 3. Extract coordinate indices
    freq_indices, time_indices = np.where(is_peak)

    # Combine into a list of (freq_bin, time_frame) tuples
    peaks = list(zip(freq_indices.tolist(), time_indices.tolist()))

    # Sort peaks chronologically by time frame (and secondarily by frequency bin)
    peaks.sort(key=lambda p: (p[1], p[0]))

    return peaks