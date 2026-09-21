import io
import time

import numpy as np
import soundfile as sf
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from audio.preprocessor import preprocess_audio
from audio.spectrogram import compute_spectrogram
from fingerprint.peaks import extract_peaks, PEAK_NEIGHBORHOOD_SIZE, MIN_AMPLITUDE_DB
from fingerprint.hasher import generate_hashes, FAN_OUT, MIN_DELTA_TIME, MAX_DELTA_TIME
from recognition.matcher import find_best_match
import config

# --------------------------------------------------------------------------
# Icons — hand-drawn inline SVGs, no emoji
# --------------------------------------------------------------------------

ICON_LOGO = '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 14 L4 10"/><path d="M8 17 L8 7"/><path d="M12 20 L12 4"/><path d="M16 17 L16 7"/><path d="M20 14 L20 10"/></svg>'
ICON_CHECK = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9.5"/><path d="M8 12.3 L10.8 15 L16 9.3"/></svg>'
ICON_ALERT = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9.5"/><line x1="12" y1="7.5" x2="12" y2="13"/><circle cx="12" cy="16.3" r="0.9" fill="currentColor" stroke="none"/></svg>'
ICON_ARTIST = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="8" r="3.4"/><path d="M5.5 20a6.5 6.5 0 0 1 13 0"/></svg>'
ICON_ALBUM = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9.5"/><circle cx="12" cy="12" r="2.3" fill="currentColor" stroke="none"/></svg>'
ICON_CLOCK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9.5"/><path d="M12 7 L12 12 L15.5 14"/></svg>'

# Shazam-style blue, used as the browser tab icon
_FAVICON = (
    "data:image/svg+xml,"
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<rect width="24" height="24" rx="6" fill="%230396FF"/>'
    '<path d="M6 14V10M9.5 17V7M13 19V5M16.5 17V7M20 14V10" '
    'stroke="white" stroke-width="1.6" stroke-linecap="round"/></svg>'
)

st.set_page_config(page_title="Mini-Shazam", page_icon=_FAVICON, layout="centered")

# --------------------------------------------------------------------------
# Styling — Shazam blue on near-black, kept deliberately simple
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    #MainMenu, footer, header { visibility: hidden; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(ellipse 800px 460px at 50% -8%, rgba(3,150,255,0.16), transparent 60%),
            #05070C;
        color: #EDEFF5;
    }
    .block-container { padding-top: 3.4rem; max-width: 560px; }

    .brand-row { display: flex; align-items: center; justify-content: center; gap: 0.55rem; margin-bottom: 0.1rem; }
    .brand-row svg { color: #4FC3FF; }
    .brand-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F5F7FF; margin: 0; }
    .brand-sub { text-align: center; color: #7C8398; font-size: 0.92rem; margin: 0.1rem 0 2.6rem 0; }

    /* Center the round recorder button and give it breathing room */
    .recorder-wrap { display: flex; justify-content: center; margin-bottom: 1rem; }
    .recorder-wrap iframe { margin: 0 auto; }

    .record-hint {
        text-align: center; color: #6B7288; font-size: 0.85rem;
        margin: 0.9rem 0 2.2rem 0;
    }

    .analyzing-wrap { display: flex; flex-direction: column; align-items: center; margin: 1.4rem 0 2rem 0; }
    .analyzing-dot-row { display: flex; gap: 8px; margin-bottom: 0.85rem; }
    .analyzing-dot { width: 9px; height: 9px; border-radius: 50%; background: #0396FF; animation: dot-bounce 1s ease-in-out infinite; }
    .analyzing-dot:nth-child(2) { animation-delay: 0.15s; }
    .analyzing-dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes dot-bounce { 0%, 60%, 100% { transform: translateY(0); opacity: 0.5; } 30% { transform: translateY(-7px); opacity: 1; } }
    .analyzing-text { color: #8890A8; font-size: 0.9rem; }

    .result-card {
        background: linear-gradient(180deg, rgba(3,150,255,0.06), rgba(255,255,255,0.015));
        border: 1px solid rgba(3,150,255,0.18);
        border-radius: 18px; padding: 1.9rem 1.7rem; margin-top: 1rem; text-align: center;
    }
    .result-badge {
        width: 48px; height: 48px; border-radius: 50%;
        background: rgba(3,150,255,0.16); color: #4FC3FF;
        display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto;
    }
    .result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: #F7F8FF; margin: 0 0 0.25rem 0; }
    .result-artist { display: flex; align-items: center; justify-content: center; gap: 0.35rem; color: #A3AABF; font-size: 0.98rem; margin-bottom: 1.2rem; }
    .result-artist svg { color: #6E7690; }
    .result-meta-row { display: flex; align-items: center; justify-content: center; gap: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.07); }
    .result-meta-item { display: flex; align-items: center; gap: 0.4rem; color: #838BA3; font-size: 0.83rem; }
    .result-meta-item svg { color: #5E6684; }
    .result-timing { text-align: center; color: #545C77; font-size: 0.76rem; margin-top: 0.9rem; }

    .alert-card {
        display: flex; align-items: center; gap: 0.85rem;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 1rem 1.2rem; margin-top: 1rem; color: #C6CAE0; font-size: 0.9rem;
    }
    .alert-card svg { color: #E0A96D; flex-shrink: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown(
    f'<div class="brand-row">{ICON_LOGO}<p class="brand-title">Mini-Shazam</p></div>'
    f'<p class="brand-sub">Tap to identify what\'s playing</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Recorder — a genuine round tap button (audio-recorder-streamlit), not a
# CSS reshaping of Streamlit's native audio_input. Records for ~7s once
# tapped, using the package's documented fixed-duration trick
# (energy_threshold forced so pause_threshold becomes the recording length).
# --------------------------------------------------------------------------

st.markdown('<div class="recorder-wrap">', unsafe_allow_html=True)
audio_bytes = audio_recorder(
    text="",
    icon_name="microphone",
    icon_size="4x",
    neutral_color="#0396FF",
    recording_color="#FF4B4B",
    energy_threshold=(-1.0, 1.0),
    pause_threshold=7.0,
    sample_rate=44100,
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="record-hint">Tap the button — it listens for 7 seconds, then identifies the track</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Processing + result
# --------------------------------------------------------------------------

if audio_bytes is not None and len(audio_bytes) > 0:
    status = st.empty()
    status.markdown(
        '<div class="analyzing-wrap">'
        '<div class="analyzing-dot-row"><div class="analyzing-dot"></div><div class="analyzing-dot"></div><div class="analyzing-dot"></div></div>'
        '<div class="analyzing-text">Matching fingerprints against your library…</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    t0 = time.time()

    # 1. Decode the recorded bytes to a NumPy array
    with io.BytesIO(audio_bytes) as buffer:
        audio_array, native_sr = sf.read(buffer, dtype="float32")

    # 2. Downmix if multi-channel
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # 3. DSP pipeline (unchanged)
    mono_audio, final_sr = preprocess_audio(audio_array, orig_sr=native_sr, target_sr=config.TARGET_SR)
    spec_db = compute_spectrogram(mono_audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    peaks = extract_peaks(spec_db, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, min_amplitude_db=MIN_AMPLITUDE_DB)
    query_hashes = generate_hashes(peaks, fan_out=FAN_OUT, min_delta_time=MIN_DELTA_TIME, max_delta_time=MAX_DELTA_TIME)

    status.empty()

    if not query_hashes:
        st.markdown(
            f'<div class="alert-card">{ICON_ALERT}<div>No prominent audio peaks detected. Try recording closer to the speaker.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # 4. Match against the fingerprint database (unchanged)
        match = find_best_match(query_hashes)
        elapsed = time.time() - t0

        if match:
            offset = match.get("offset_seconds", 0) or 0
            mins, secs = int(offset // 60), int(offset % 60)
            duration = match.get("duration_sec")
            album = match.get("album")

            meta_items = f'<div class="result-meta-item">{ICON_CLOCK}<span>~{mins:02d}:{secs:02d} cue</span></div>'
            if duration:
                meta_items += f'<div class="result-meta-item">{ICON_CLOCK}<span>{duration:.0f}s track</span></div>'
            if album:
                meta_items += f'<div class="result-meta-item">{ICON_ALBUM}<span>{album}</span></div>'

            result_html = (
                f'<div class="result-card">'
                f'<div class="result-badge">{ICON_CHECK}</div>'
                f'<p class="result-title">{match["title"]}</p>'
                f'<div class="result-artist">{ICON_ARTIST}<span>{match["artist"]}</span></div>'
                f'<div class="result-meta-row">{meta_items}</div>'
                f'</div>'
                f'<p class="result-timing">Identified in {elapsed:.2f}s</p>'
            )
            st.markdown(result_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert-card">{ICON_ALERT}<div>No confident match found. Try playing louder or holding the mic closer.</div></div>',
                unsafe_allow_html=True,
            )