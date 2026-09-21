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

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
    "%3Ccircle cx='32' cy='32' r='30' fill='%230A84FF'/%3E"
    "%3Ccircle cx='32' cy='32' r='20' fill='none' stroke='white' stroke-width='4'/%3E"
    "%3Ccircle cx='32' cy='32' r='6' fill='white'/%3E"
    "%3C/svg%3E"
)

st.set_page_config(
    page_title="Sonic ID — Audio Recognition",
    page_icon=FAVICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────
# ICONS (inline SVG — no emoji, single accent color, stroke-based / Shazam-ish)
# ──────────────────────────────────────────────────────────────────────────
ICON_LOGO = """
<svg width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="36" cy="36" r="34" fill="url(#logoGrad)"/>
  <circle cx="36" cy="36" r="24" stroke="rgba(255,255,255,0.55)" stroke-width="2.5" fill="none"/>
  <circle cx="36" cy="36" r="14" stroke="rgba(255,255,255,0.85)" stroke-width="2.5" fill="none"/>
  <circle cx="36" cy="36" r="5" fill="#FFFFFF"/>
  <defs>
    <linearGradient id="logoGrad" x1="0" y1="0" x2="72" y2="72" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#38B6FF"/>
      <stop offset="1" stop-color="#0033A0"/>
    </linearGradient>
  </defs>
</svg>
"""

ICON_MUSIC_NOTE = """
<svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M9 18V5l11-2v13" stroke="#0A84FF" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="6" cy="18" r="3" stroke="#0A84FF" stroke-width="1.7"/>
  <circle cx="17" cy="16" r="3" stroke="#0A84FF" stroke-width="1.7"/>
</svg>
"""

ICON_ARTIST = """
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="8" r="3.4" stroke="#9FB8D9" stroke-width="1.6"/>
  <path d="M5 20c1.2-3.6 4-5.4 7-5.4s5.8 1.8 7 5.4" stroke="#9FB8D9" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

ICON_CLOCK = """
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="8.4" stroke="#9FB8D9" stroke-width="1.6"/>
  <path d="M12 7.5V12l3 2" stroke="#9FB8D9" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_PLAY = """
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M6 4.8v14.4L19 12 6 4.8Z" fill="#9FB8D9"/>
</svg>
"""

ICON_WARNING = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 3.5 2 20.5h20L12 3.5Z" stroke="#FFB020" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M12 9.5v5" stroke="#FFB020" stroke-width="1.6" stroke-linecap="round"/>
  <circle cx="12" cy="17.2" r="0.9" fill="#FFB020"/>
</svg>
"""

ICON_ERROR = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="9" stroke="#FF5C5C" stroke-width="1.6"/>
  <path d="M9 9l6 6M15 9l-6 6" stroke="#FF5C5C" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

# ──────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, header, footer { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }

    .stApp {
        background: radial-gradient(ellipse 120% 80% at 50% -10%, #10306B 0%, #050A16 55%, #030509 100%);
        color: #EAF1FB;
    }

    .block-container { padding-top: 3rem; max-width: 560px; }

    /* ── Header ───────────────────────────────────────────── */
    .app-header { text-align: center; margin-bottom: 0.4rem; }
    .app-header .logo-wrap { display: flex; justify-content: center; margin-bottom: 0.9rem; }
    .app-title {
        font-weight: 800; font-size: 2.1rem; letter-spacing: 0.06em;
        background: linear-gradient(90deg, #7FD0FF, #FFFFFF 45%, #7FD0FF);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        margin: 0;
    }
    .app-subtitle {
        color: #8FA6C9; font-size: 0.95rem; font-weight: 500;
        margin-top: 0.35rem; letter-spacing: 0.01em;
    }

    hr { border-color: rgba(255,255,255,0.08); margin: 1.6rem 0; }

    /* ── Audio input → round Shazam-style capture button ────── */
    div[data-testid="stAudioInput"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(120,170,255,0.18);
        border-radius: 28px;
        padding: 1.6rem 1.2rem;
        box-shadow: 0 0 60px rgba(10,132,255,0.08), inset 0 0 0 1px rgba(255,255,255,0.02);
    }
    div[data-testid="stAudioInput"] button {
        background: radial-gradient(circle at 35% 30%, #4FC3FF, #0A84FF 55%, #0033A0 100%) !important;
        border: none !important;
        box-shadow: 0 0 0 6px rgba(10,132,255,0.14), 0 8px 24px rgba(10,132,255,0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stAudioInput"] button:hover {
        transform: scale(1.04);
        box-shadow: 0 0 0 9px rgba(10,132,255,0.18), 0 10px 28px rgba(10,132,255,0.45) !important;
    }
    div[data-testid="stAudioInput"] svg { fill: #FFFFFF !important; }
    div[data-testid="stAudioInput"] p, div[data-testid="stAudioInput"] span { color: #B9CBE8 !important; }

    /* ── Listening / analyzing animation ─────────────────────── */
    .pulse-wrap {
        display: flex; flex-direction: column; align-items: center;
        padding: 2.2rem 0 1.6rem 0;
    }
    .pulse-rings {
        position: relative; width: 120px; height: 120px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 1.3rem;
    }
    .pulse-ring {
        position: absolute; border-radius: 50%;
        border: 2px solid rgba(10,132,255,0.55);
        width: 100%; height: 100%;
        animation: pulseGrow 1.8s cubic-bezier(0.2,0.6,0.35,1) infinite;
    }
    .pulse-ring.r2 { animation-delay: 0.6s; }
    .pulse-ring.r3 { animation-delay: 1.2s; }
    .pulse-core {
        width: 52px; height: 52px; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #4FC3FF, #0A84FF 55%, #0033A0 100%);
        box-shadow: 0 0 30px rgba(10,132,255,0.65);
        z-index: 2;
    }
    @keyframes pulseGrow {
        0%   { transform: scale(0.35); opacity: 0.9; }
        100% { transform: scale(1.15); opacity: 0; }
    }
    .pulse-label {
        color: #9FC2FF; font-weight: 600; font-size: 0.95rem;
        letter-spacing: 0.02em; text-align: center;
    }
    /* ── Result card ──────────────────────────────────────────── */
    .result-card {
        margin-top: 1.6rem;
        background: linear-gradient(165deg, rgba(20,50,100,0.55), rgba(6,12,26,0.75));
        border: 1px solid rgba(120,170,255,0.22);
        border-radius: 22px;
        padding: 1.8rem 1.7rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        animation: cardIn 0.45s ease;
    }
    @keyframes cardIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-tag {
        display: inline-flex; align-items: center; gap: 0.4rem;
        color: #6FE3A8; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .result-tag .dot {
        width: 7px; height: 7px; border-radius: 50%; background: #6FE3A8;
        box-shadow: 0 0 8px #6FE3A8;
    }
    .result-header { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1.3rem; }
    .result-icon {
        width: 54px; height: 54px; border-radius: 14px;
        background: rgba(10,132,255,0.12);
        border: 1px solid rgba(10,132,255,0.3);
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .result-title { font-size: 1.45rem; font-weight: 800; color: #FFFFFF; line-height: 1.25; margin: 0; }
    .result-artist { font-size: 1rem; color: #9FB8D9; font-weight: 500; margin-top: 0.2rem; }

    .result-meta { display: flex; gap: 0.9rem; flex-wrap: wrap; }
    .meta-chip {
        display: flex; align-items: center; gap: 0.45rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.5rem 0.85rem;
        font-size: 0.85rem; color: #C7D6EE; font-weight: 500;
    }

    .status-box {
        display: flex; align-items: flex-start; gap: 0.7rem;
        border-radius: 16px; padding: 1rem 1.1rem; margin-top: 1.4rem;
        font-size: 0.92rem; line-height: 1.4;
    }
    .status-box.warn { background: rgba(255,176,32,0.08); border: 1px solid rgba(255,176,32,0.25); color: #FFD79A; }
    .status-box.err  { background: rgba(255,92,92,0.08); border: 1px solid rgba(255,92,92,0.25); color: #FFB3B3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-header">
        <div class="logo-wrap">{ICON_LOGO}</div>
        <p class="app-title">SONIC&nbsp;ID</p>
        <p class="app-subtitle">Identify any track from your indexed catalog — instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# CAPTURE
# ──────────────────────────────────────────────────────────────────────────
audio_file = st.audio_input("Record a 5–7 second clip of the song playing")

result_slot = st.empty()

if audio_file is not None:
    audio_bytes = audio_file.read()

    listening_slot = st.empty()
    listening_slot.markdown(
        """
        <div class="pulse-wrap">
            <div class="pulse-rings">
                <div class="pulse-ring r1"></div>
                <div class="pulse-ring r2"></div>
                <div class="pulse-ring r3"></div>
                <div class="pulse-core"></div>
            </div>
            <div class="pulse-label">Matching audio fingerprint…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        listening_slot.empty()
        result_slot.markdown(
            f"""
            <div class="status-box warn">
                {ICON_WARNING}
                <div><strong>No prominent audio peaks detected.</strong><br/>Try recording closer to the speaker, in a quieter environment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # 4. Query Neon DB Matcher
        match = find_best_match(query_hashes)
        elapsed = time.time() - t0

        listening_slot.empty()

        if match:
            mins = int(match["offset_seconds"] // 60)
            secs = int(match["offset_seconds"] % 60)

            result_slot.markdown(
                f"""
                <div class="result-card">
                    <div class="result-tag"><span class="dot"></span>Match found · {elapsed:.2f}s</div>
                    <div class="result-header">
                        <div class="result-icon">{ICON_MUSIC_NOTE}</div>
                        <div>
                            <p class="result-title">{match['title']}</p>
                            <div class="result-artist">{match['artist']}</div>
                        </div>
                    </div>
                    <div class="result-meta">
                        <div class="meta-chip">{ICON_PLAY}&nbsp;Cue&nbsp;{mins:02d}:{secs:02d}</div>
                        <div class="meta-chip">{ICON_CLOCK}&nbsp;{match['duration_sec']:.1f}s duration</div>
                        <div class="meta-chip">{ICON_ARTIST}&nbsp;{match['artist']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            result_slot.markdown(
                f"""
                <div class="status-box err">
                    {ICON_ERROR}
                    <div><strong>No confident match found.</strong><br/>Try playing the track louder or holding the microphone closer to the source.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )