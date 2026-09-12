import os
from dotenv import load_dotenv

load_dotenv()

# Database Connection Settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "minishazam")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")

# Core Audio Signal Processing Constants
TARGET_SR = 44100      # Target audio sample rate (Hz)
N_FFT = 4096           # FFT window length (samples)
HOP_LENGTH = 512       # Hop length between successive STFT columns