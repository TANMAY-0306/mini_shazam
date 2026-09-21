import base64
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

# --------------------------------------------------------------------------
# Icons (hand-drawn inline SVGs — no emoji, no external icon font dependency)
# --------------------------------------------------------------------------

ICON_LOGO = """
<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <path d="M4 14 L4 10" /><path d="M8 17 L8 7" />
  <path d="M12 20 L12 4" /><path d="M16 17 L16 7" />
  <path d="M20 14 L20 10" />
</svg>
"""

ICON_MIC = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
  <rect x="9" y="2" width="6" height="12" rx="3" />
  <path d="M5 11a7 7 0 0 0 14 0" />
  <line x1="12" y1="18" x2="12" y2="22" />
  <line x1="8" y1="22" x2="16" y2="22" />
</svg>
"""

ICON_CHECK = """
<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="9.5" />
  <path d="M8 12.3 L10.8 15 L16 9.3" />
</svg>
"""

ICON_ALERT = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="9.5" />
  <line x1="12" y1="7.5" x2="12" y2="13" />
  <circle cx="12" cy="16.3" r="0.9" fill="currentColor" stroke="none" />
</svg>
"""

ICON_ARTIST = """
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="8" r="3.4" />
  <path d="M5.5 20a6.5 6.5 0 0 1 13 0" />
</svg>
"""

ICON_ALBUM = """
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="9.5" />
  <circle cx="12" cy="12" r="2.3" fill="currentColor" stroke="none" />
</svg>
"""

ICON_CLOCK = """
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="9.5" />
  <path d="M12 7 L12 12 L15.5 14" />
</svg>
"""


def _favicon_data_uri() -> str:
    """Build the browser tab icon from the same mark used in the header — no emoji."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="%23EDEBFF" stroke-width="2" stroke-linecap="round">'
        '<rect width="24" height="24" rx="6" fill="%236C5CE7"/>'
        '<path d="M6 14V10M9.5 17V7M13 19V5M16.5 17V7M20 14V10" '
        'stroke="white" stroke-width="1.6"/></svg>'
    )
    return "data:image/svg+xml," + svg


st.set_page_config(
    page_title="Sonic — Song Recognition",
    page_icon=_favicon_data_uri(),
    layout="centered",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    #MainMenu, footer, header { visibility: hidden; }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(ellipse 900px 500px at 50% -10%, rgba(108,92,231,0.20), transparent 60%),
            #05060B;
        color: #EDEDF5;
    }

    .block-container { padding-top: 3.5rem; max-width: 640px; }

    /* ---------- Header ---------- */
    .brand-row {
        display: flex; align-items: center; justify-content: center;
        gap: 0.6rem; margin-bottom: 0.15rem;
    }
    .brand-row svg { color: #8B7CF6; }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em;
        color: #F5F4FF; margin: 0;
    }
    .brand-sub {
        text-align: center; color: #8188A6; font-size: 0.95rem;
        margin-top: 0.1rem; margin-bottom: 2.4rem;
    }

    /* ---------- Record shell ---------- */
    .record-shell {
        position: relative;
        width: 220px; height: 220px;
        margin: 0 auto 0.6rem auto;
        display: flex; align-items: center; justify-content: center;
    }
    .record-ring {
        position: absolute; border-radius: 50%;
        border: 1px solid rgba(139,124,246,0.25);
    }
    .ring-1 { width: 220px; height: 220px; }
    .ring-2 { width: 178px; height: 178px; }

    .pulse-ring {
        position: absolute; border-radius: 50%;
        border: 1.5px solid rgba(139,124,246,0.55);
        animation: pulse-out 1.8s cubic-bezier(0.2, 0.7, 0.4, 1) infinite;
    }
    .pulse-ring.delay { animation-delay: 0.6s; }
    @keyframes pulse-out {
        0%   { width: 130px; height: 130px; opacity: 0.75; }
        100% { width: 220px; height: 220px; opacity: 0; }
    }

    .record-core {
        position: relative; z-index: 2;
        width: 128px; height: 128px; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #8B7CF6, #6C5CE7 55%, #4B3FB0 100%);
        box-shadow: 0 8px 28px rgba(108,92,231,0.45), inset 0 1px 1px rgba(255,255,255,0.25);
    }

    /* Nudge Streamlit's native audio widget to sit inside the round shell,
       centered and de-chromed. Selector targets Streamlit's testid; if a
       future Streamlit version renames it, re-inspect via devtools and
       update this selector. */
    div[data-testid="stAudioInput"] {
        display: flex; justify-content: center;
        margin-top: -190px; /* overlay onto .record-shell above */
        position: relative; z-index: 3;
        transform: scale(1.35);
    }
    div[data-testid="stAudioInput"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stAudioInput"] button {
        color: #FFFFFF !important;
    }

    .record-hint {
        text-align: center; color: #6E7492; font-size: 0.85rem;
        margin-top: 0.4rem; margin-bottom: 2.2rem;
        display: flex; align-items: center; justify-content: center; gap: 0.4rem;
    }
    .record-hint svg { color: #6E7492; }

    /* ---------- Analyzing state ---------- */
    .analyzing-wrap { display: flex; flex-direction: column; align-items: center; margin: 1.6rem 0 2.2rem 0; }
    .analyzing-dot-row { display: flex; gap: 8px; margin-bottom: 0.9rem; }
    .analyzing-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: #8B7CF6; animation: dot-bounce 1s ease-in-out infinite;
    }
    .analyzing-dot:nth-child(2) { animation-delay: 0.15s; }
    .analyzing-dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes dot-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-7px); opacity: 1; }
    }
    .analyzing-text { color: #9096B5; font-size: 0.92rem; letter-spacing: 0.01em; }

    /* ---------- Result card ---------- */
    .result-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 2rem 1.8rem;
        margin-top: 1.2rem;
        text-align: center;
    }
    .result-badge {
        width: 52px; height: 52px; border-radius: 50%;
        background: rgba(108,92,231,0.18); color: #9C8CFF;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.1rem auto;
    }
    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem; font-weight: 700; color: #F7F6FF;
        margin: 0 0 0.3rem 0; line-height: 1.25;
    }
    .result-artist {
        display: flex; align-items: center; justify-content: center; gap: 0.35rem;
        color: #A9AEC9; font-size: 1rem; margin-bottom: 1.3rem;
    }
    .result-artist svg { color: #7B81A0; }

    .result-meta-row {
        display: flex; align-items: center; justify-content: center;
        gap: 1.6rem; padding-top: 1.1rem;
        border-top: 1px solid rgba(255,255,255,0.07);
    }
    .result-meta-item {
        display: flex; align-items: center; gap: 0.4rem;
        color: #8991B4; font-size: 0.85rem;
    }
    .result-meta-item svg { color: #6E749A; }

    .result-timing {
        text-align: center; color: #5C6284; font-size: 0.78rem; margin-top: 1rem;
    }

    /* ---------- No-match / alert card ---------- */
    .alert-card {
        display: flex; align-items: center; gap: 0.9rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.1rem 1.3rem;
        margin-top: 1.2rem; color: #C6CAE0; font-size: 0.92rem;
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
    f"""
    <div class="brand-row">{ICON_LOGO}<p class="brand-title">Sonic</p></div>
    <p class="brand-sub">Identify any track from your library in seconds</p>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Record shell (decorative rings behind the native audio widget)
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="record-shell">
        <div class="record-ring ring-1"></div>
        <div class="record-ring ring-2"></div>
        <div class="pulse-ring"></div>
        <div class="pulse-ring delay"></div>
        <div class="record-core"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

audio_file = st.audio_input("Record", label_visibility="collapsed")

st.markdown(
    f"""<p class="record-hint">{ICON_MIC} Tap the circle and play a 5–7 second clip</p>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Processing + result
# --------------------------------------------------------------------------

if audio_file is not None:
    status = st.empty()
    status.markdown(
        """
        <div class="analyzing-wrap">
            <div class="analyzing-dot-row">
                <div class="analyzing-dot"></div>
                <div class="analyzing-dot"></div>
                <div class="analyzing-dot"></div>
            </div>
            <div class="analyzing-text">Matching fingerprints against your library…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t0 = time.time()
    audio_bytes = audio_file.read()

    # 1. Read byte stream to NumPy array
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
            f"""
            <div class="alert-card">{ICON_ALERT}
                <div>No prominent audio peaks detected. Try recording closer to the speaker.</div>
            </div>
            """,
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

            meta_items = f"""
                <div class="result-meta-item">{ICON_CLOCK}<span>~{mins:02d}:{secs:02d} cue</span></div>
            """
            if duration:
                meta_items += f"""
                <div class="result-meta-item">{ICON_CLOCK}<span>{duration:.0f}s track</span></div>
                """
            if album:
                meta_items += f"""
                <div class="result-meta-item">{ICON_ALBUM}<span>{album}</span></div>
                """

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-badge">{ICON_CHECK}</div>
                    <p class="result-title">{match['title']}</p>
                    <div class="result-artist">{ICON_ARTIST}<span>{match['artist']}</span></div>
                    <div class="result-meta-row">{meta_items}</div>
                </div>
                <p class="result-timing">Identified in {elapsed:.2f}s</p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-card">{ICON_ALERT}
                    <div>No confident match found. Try playing louder or holding the mic closer.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )