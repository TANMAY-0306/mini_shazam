import io
import time
import numpy as np
import soundfile as sf
import streamlit as st

from audio.preprocessor import preprocess_audio
from audio.spectrogram import compute_spectrogram
from fingerprint.peaks import extract_peaks, PEAK_NEIGHBORHOOD_SIZE, MIN_AMPLITUDE_DB
from fingerprint.hasher import generate_hashes, FAN_OUT, MIN_DELTA_TIME, MAX_DELTA_TIME
from recognition.matcher import find_best_match
import config

st.set_page_config(
    page_title="Mini Shazam",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Mini Shazam")
st.caption("Identify any song from your indexed catalog in real time.")

st.markdown("---")

# Native Streamlit audio recording component
audio_file = st.audio_input("Record a 5–7 second clip of the song playing:")

if audio_file is not None:
    audio_bytes = audio_file.read()

    with st.spinner("Analyzing audio fingerprints against Neon DB..."):
        t0 = time.time()

        # 1. Read byte stream to NumPy array
        with io.BytesIO(audio_bytes) as buffer:
            audio_array, native_sr = sf.read(buffer, dtype="float32")

        # 2. Downmix if multi-channel
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        # 3. DSP Pipeline
        mono_audio, final_sr = preprocess_audio(audio_array, orig_sr=native_sr, target_sr=config.TARGET_SR)
        spec_db = compute_spectrogram(mono_audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        peaks = extract_peaks(spec_db, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, min_amplitude_db=MIN_AMPLITUDE_DB)
        query_hashes = generate_hashes(peaks, fan_out=FAN_OUT, min_delta_time=MIN_DELTA_TIME, max_delta_time=MAX_DELTA_TIME)

        if not query_hashes:
            st.warning("No prominent audio peaks detected. Try recording closer to the speaker.")
        else:
            # 4. Query Neon DB Matcher
            match = find_best_match(query_hashes)
            elapsed = time.time() - t0

            if match:
                st.success("🎉 Song Identified!")

                mins = int(match["offset_seconds"] // 60)
                secs = int(match["offset_seconds"] % 60)

                st.markdown(f"""
                ### **{match['title']}**
                * **Artist:** {match['artist']}
                * **Confidence Score:** `{match['confidence_score']}` matching landmarks
                * **Playback Cue:** `~{mins:02d}:{secs:02d}`
                * **Track Duration:** `{match['duration_sec']:.1f}s`
                *(Processed in {elapsed:.2f} seconds)*
                """)
            else:
                st.error("No confident match found. Try playing louder or holding the mic closer.")