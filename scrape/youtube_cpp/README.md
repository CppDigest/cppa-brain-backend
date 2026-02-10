# YouTube C++ Videos Scraper

Scrapes YouTube C++ videos using YouTube Data API v3 and yt-dlp.

## Download Modes

| CLI flag        | Collects                         | Skips if exists           |
| --------------- | --------------------------------- | ------------------------- |
| (default)       | JSON metadata + transcript (VTT)  | JSON + transcript         |
| `--need-video`  | JSON + transcript + video (MP4)   | JSON + transcript + video |

- Default: discover via search and download **transcripts** (and save metadata JSON).
- `--need-video`: also download **videos** (MP4). Use with `--retry-downloads` to fill missing from existing `data/json`.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key: copy .env.template to .env and set YOUTUBE_API_KEY, or:
export YOUTUBE_API_KEY="your-key"

# Discover and download transcripts (default)
python scraper.py

# Download videos (for items in data/json)
python scraper.py --need-video --retry-downloads

# Limit search terms
python scraper.py --limit 10
```

## Prerequisites

- **YouTube Data API v3 key** ([Get here](https://console.cloud.google.com/))
- **yt-dlp** and **yt-dlp-ejs** (for transcript/full modes)
- **JavaScript runtime** for YouTube: **Deno** (recommended) or **Node.js** — required so yt-dlp can solve YouTube's n-challenge and get video formats. Without it you get "Only images are available" / "Requested format is not available".
- **FFmpeg** (for full mode if merging formats)
- **Cookies** (for transcript or video downloads): if you see *"Sign in to confirm you're not a bot"*, see [Cookies (avoid bot detection)](#cookies-avoid-bot-detection) below.

## Cookies (avoid bot detection)

If yt-dlp fails with *"Sign in to confirm you're not a bot"*, pass YouTube cookies via one of these methods.

### Option 1: Use browser cookies (easiest on Mac)

1. Log into [YouTube](https://www.youtube.com) in **Safari** or **Chrome**.
2. In `.env` (or as environment variables), set:
   - **Safari:** `YT_COOKIES_BROWSER=safari`
   - **Chrome:** `YT_COOKIES_BROWSER=chrome`
3. Run the scraper. No cookie file needed; cookies are read from the browser.

### Option 2: Use a cookies file

1. Install a cookies-export extension (e.g. **Get cookies.txt** for Chrome, or **cookies.txt** for Safari).
2. Open [youtube.com](https://www.youtube.com) while logged in and export cookies in **Netscape** format (e.g. `cookies.txt`).
3. In `.env`, set the path: `YT_COOKIES_FILE=/path/to/cookies.txt`
4. Run the scraper.

Cookie files expire; re-export if the bot error returns.

## Configuration (yt_config.py)

### Key Settings

- `CONCURRENT_FRAGMENTS`: `1` (sequential, prevents 403 errors)
- `DOWNLOAD_SLEEP_INTERVAL`: `3.0` (seconds between requests)
- `VIDEO_DOWNLOAD_DELAY`: `3.0` (seconds between videos)

### Search Terms (search_terms.py)

- `C_PLUS_PLUS_CREATORS`: Creator names to search
- `C_PLUS_PLUS_CHANNELS`: Channel IDs
- `C_PLUS_PLUS_PLAYLISTS`: Playlist IDs
- `SEARCH_TERMS`: Generic search terms

### Filters

- `MIN_DURATION_SECONDS`: `60` (1 minute minimum)
- `MAX_DURATION_SECONDS`: `14400` (4 hours maximum)
- `PUBLISHED_AFTER`: `"2022-01-01T00:00:00Z"`

## Output Structure

```
data/
├── json/{video_id}.json              # Metadata (all runs)
├── transcripts/{video_id}.en.vtt     # Transcripts (default or --need-video)
├── videos/{video_id}.mp4             # Videos (only with --need-video)
├── progress_youtube_cpp.json         # Progress tracking
└── logs/youtube_cpp_scraper.log      # Log file
```

## Notes

- Duplicate videos automatically skipped
- Progress saved periodically (safe to interrupt)
- Videos >4 hours skipped when using `--need-video`
- Video format: `best[ext=mp4]/best` to reduce 403 errors
- Minimum 3s delay enforced in code (even if config is lower)
- Default run: search + transcript download; use `--retry-downloads` to only fill missing content from existing `data/json`
