"""
Configuration settings for YouTube C++ videos scraper
"""

import os

# YouTube Data API v3 Settings
# Set your API key here or use environment variable YOUTUBE_API_KEY (recommended for security)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Rate Limiting
# YouTube Data API v3 quota: 10,000 units/day (default)
# Search: 100 units per request
# Video details: 1 unit per video
MIN_DELAY_SECONDS = 0.1  # Small delay between API calls
RECOMMENDED_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 2.0

# Request Settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier

# Output Directories
OUTPUT_DIR = "data/youtube_cpp"

# Progress Tracking
PROGRESS_FILE = "data/progress_youtube_cpp.json"
CHECKPOINT_INTERVAL = 50  # Save progress every N videos

# Search Parameters
MAX_RESULTS_PER_SEARCH = 50  # YouTube API max is 50
SEARCH_PART = "id,snippet"  # Parts to retrieve in search

# Video Details Parameters
VIDEO_PART = "snippet,contentDetails,statistics"  # Parts to retrieve for video details
PUBLISHED_AFTER = "2022-01-01T00:00:00Z"
VIDEO_CATEGORY_ID = None  # Optional: YouTube category id filter

# Metadata JSON directory: one file per video, filename = {video_id}.json
METADATA_JSON_DIR = "data/json"

# Video Download Settings (yt-dlp)
VIDEO_DOWNLOAD_DIR = "data/videos"  # Directory for downloaded videos
TRANSCRIPT_DIR = "data/transcripts"  # Directory for transcripts

# Cookies (avoid "Sign in to confirm you're not a bot")
# Option 1: Use browser cookies (easiest on Mac). Set to browser name, e.g. "safari" or "chrome".
# You must be logged into YouTube in that browser.
YT_COOKIES_BROWSER = os.getenv("YT_COOKIES_BROWSER", "").strip().lower() or None
# Option 2: Path to a Netscape-format cookies file (e.g. from an extension like "Get cookies.txt").
YT_COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "").strip() or None
VIDEO_FORMAT = (
    "bv*[height<=720]+ba/b[height<=720]"
    # bv* = best video, ba = best audio, b = best combined (fallback)
    # This format downloads video and audio separately then merges (faster)
    # For maximum speed, you can use: "bv*+ba/b" (without height limit)
)
AUDIO_ONLY = False  # Download audio only (smaller files)
SUBTITLE_LANGUAGES = ["en", "en-US", "en-GB"]  # Preferred subtitle languages
AUTO_SUBTITLES = True  # Use auto-generated subtitles if manual not available
SKIP_EXISTING = True  # Skip videos that already exist

# Rate Limiting for Video Downloads (to avoid YouTube rate limits)
VIDEO_DOWNLOAD_DELAY = 3.0
# Seconds to wait between video downloads (recommended: 3-5 seconds)
TRANSCRIPT_DOWNLOAD_DELAY = 0.5
# Seconds to wait between transcript downloads
VIDEO_DOWNLOAD_SLEEP_INTERVAL = 1
# Sleep interval for yt-dlp video downloads (seconds between requests)
DOWNLOAD_SLEEP_INTERVAL = 3
# Sleep interval for yt-dlp downloads (increased from 0.5 to 3.0 to avoid 403 errors)
# IMPORTANT: If you get "HTTP 403: Forbidden" errors, increase this to 5.0 or higher

# Rate limit handling
RATE_LIMIT_BACKOFF_MULTIPLIER = 2.0  # Multiply delay by this on each retry
RATE_LIMIT_INITIAL_WAIT = 5.0  # Initial wait time (seconds) when rate limited

# Concurrent Transcript Downloads (for even faster downloads)
# Note: Enabling this will download multiple transcripts in parallel
# This can significantly speed up downloads but may trigger rate limits
# Start with 1-2, increase gradually if no rate limiting occurs
CONCURRENT_TRANSCRIPT_DOWNLOADS = 1  # Number of transcripts to download in parallel
# Set to 1 for sequential (safest), 2-5 for parallel (faster but riskier)

# Download Speed Optimization
CONCURRENT_FRAGMENTS = 1
# Number of fragments to download in parallel (set to 1 for sequential - most conservative)
# IMPORTANT: Higher values (2-8) are faster but trigger "HTTP 403: Forbidden" errors frequently
# Keep at 1 for maximum reliability, or try 2 if downloads are stable
# Note: YouTube aggressively rate-limits parallel fragment downloads

# Data Validation
MIN_DURATION_SECONDS = 60  # Minimum video duration in seconds
MAX_DURATION_SECONDS = 4 * 60 * 60  # Maximum video duration in seconds (4 hours)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "data/logs/youtube_cpp_scraper.log"
