import hashlib
from typing import List, Tuple

# Pairing hyperparameters
FAN_OUT: int = 10          # Maximum number of target peaks paired per anchor
MIN_DELTA_TIME: int = 5    # Minimum forward time offset in frames (~58 ms)
MAX_DELTA_TIME: int = 200  # Maximum forward time offset in frames (~2.32 s)


def generate_hashes(
    peaks: List[Tuple[int, int]],
    fan_out: int = FAN_OUT,
    min_delta_time: int = MIN_DELTA_TIME,
    max_delta_time: int = MAX_DELTA_TIME
) -> List[Tuple[str, int]]:
    """
    Pairs constellation peaks within a forward target zone and produces SHA-1 hashes.

    Args:
        peaks: List of (freq_bin, t_time_frame) tuples, sorted chronologically by time.
        fan_out: Maximum target peaks paired per anchor peak.
        min_delta_time: Minimum forward frame difference (t2 - t1).
        max_delta_time: Maximum forward frame difference (t2 - t1).

    Returns:
        List[Tuple[str, int]]: List of (hash_digest, t1_time_frame) tuples.
    """
    hashes: List[Tuple[str, int]] = []
    num_peaks = len(peaks)

    for i in range(num_peaks):
        f1, t1_time_frame = peaks[i]

        paired_count = 0
        for j in range(i + 1, num_peaks):
            f2, t2_time_frame = peaks[j]
            delta_t = t2_time_frame - t1_time_frame

            # Skip if target peak is too close to anchor
            if delta_t < min_delta_time:
                continue

            # Target zone exceeded: peaks are time-sorted, so break early
            if delta_t > max_delta_time:
                break

            # Construct standardized hash token: f1|f2|delta_t
            token = f"{f1}|{f2}|{delta_t}".encode("utf-8")
            hash_digest = hashlib.sha1(token).hexdigest()[:20]

            # Store as (hash, t1_time_frame)
            hashes.append((hash_digest, t1_time_frame))
            paired_count += 1

            if paired_count >= fan_out:
                break

    return hashes