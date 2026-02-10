"""
Main scraper implementation for YouTube C++ videos
"""

from pathlib import Path

# Load .env from project directory (optional; requires python-dotenv)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import yt_config
from search_terms import (
    SEARCH_TERMS,
    C_PLUS_PLUS_CHANNELS,
    C_PLUS_PLUS_CREATORS,
    C_PLUS_PLUS_PLAYLISTS,
)
from utils import (
    rate_limit,
    retry,
    get_api_key,
    ensure_directories,
    save_progress,
    load_progress,
    format_duration,
    parse_duration,
    format_video_data,
)

# Ensure directories exist before setting up logging
ensure_directories()

# Set up logging
log_file_path = Path(yt_config.LOG_FILE)
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, yt_config.LOG_LEVEL),
    format=yt_config.LOG_FORMAT,
    handlers=[logging.FileHandler(yt_config.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class YouTubeCppScraper:
    """
    Scraper for YouTube C++ videos using YouTube Data API v3.
    """

    def __init__(self, need_video: bool = False):
        """
        Initialize scraper.

        Args:
            need_video: If True, download video (MP4); if False, download transcript only.
        """
        
        self.visited_videos: Set[str] = self.get_existed_videos()
        self.progress = load_progress(yt_config.PROGRESS_FILE)
        self.quota_used = self.progress.get("quota_used", 0)

        self.missing_videos: List[Dict[str, str]] = []
        self.video_ids = []
        self.need_video = need_video

        self.ydl_opts = None
        self._setup_ytdlp()

        ensure_directories()

        logger.info(
            "Scraper initialized: mode=%s, delay=%s",
            "need_video" if self.need_video else "need_transcript",
            yt_config.RECOMMENDED_DELAY_SECONDS,
        )
        if self.need_video:
            logger.info("Video download enabled: %s", yt_config.VIDEO_DOWNLOAD_DIR)
        else:
            logger.info("Transcript download enabled: %s", yt_config.TRANSCRIPT_DIR)

    def get_existed_videos(self) -> Set[str]:
        """Load set of video IDs that already have output (for skip logic)."""

        if self.need_video:
            video_dir = Path(yt_config.VIDEO_DOWNLOAD_DIR)
            if video_dir.exists():
                return {f.stem for f in video_dir.glob("*.mp4")}

        if not self.need_video:
            trans_dir = Path(yt_config.TRANSCRIPT_DIR)
            if trans_dir.exists():
                return {f.stem.split(".")[0] for f in trans_dir.glob("*.vtt")}

        json_dir = Path(yt_config.METADATA_JSON_DIR)
        if json_dir.exists():
            return {f.stem for f in json_dir.glob("*.json")}
        return set()

    def _save_video_metadata(self, video_id: str, data: Dict) -> None:
        """Save video metadata as JSON; skip if file already exists."""

        path = Path(yt_config.METADATA_JSON_DIR) / f"{video_id}.json"

        if path.exists() and yt_config.SKIP_EXISTING:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("Saved metadata: %s", path)

    def _setup_ytdlp(self):
        """Setup yt-dlp options for video and transcript download."""

        if self.need_video:
            Path(yt_config.VIDEO_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

            outtmpl = str(Path(yt_config.VIDEO_DOWNLOAD_DIR) / "%(id)s.%(ext)s")
            self.ydl_opts = {
                "format": "best[ext=mp4]/best",
                "skip_download": False,
                "concurrent_fragment_downloads": yt_config.CONCURRENT_FRAGMENTS,
            }
        else:
            outtmpl = str(Path(yt_config.TRANSCRIPT_DIR) / "%(id)s.%(ext)s")
            self.ydl_opts = {
                "skip_download": True,
            }

        self.ydl_opts.update(
            {
                "check-formats": True,
                "force_ipv4": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": yt_config.SUBTITLE_LANGUAGES,
                "subtitlesformat": "vtt",
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "sleep_interval": yt_config.DOWNLOAD_SLEEP_INTERVAL,
                "sleep_interval_requests": yt_config.DOWNLOAD_SLEEP_INTERVAL,
                "sleep_interval_subtitles": yt_config.DOWNLOAD_SLEEP_INTERVAL,
            }
        )
        # Use cookies to avoid "Sign in to confirm you're not a bot" (for both video and transcript)
        if yt_config.YT_COOKIES_FILE and Path(yt_config.YT_COOKIES_FILE).exists():
            self.ydl_opts["cookiefile"] = yt_config.YT_COOKIES_FILE
            logger.info("Using cookies from file: %s", yt_config.YT_COOKIES_FILE)
        elif yt_config.YT_COOKIES_BROWSER:
            self.ydl_opts["cookiesfrombrowser"] = (yt_config.YT_COOKIES_BROWSER,)
            logger.info("Using cookies from browser: %s", yt_config.YT_COOKIES_BROWSER)

    @retry(
        max_attempts=yt_config.MAX_RETRIES,
        backoff=yt_config.RETRY_BACKOFF_FACTOR,
        exceptions=(HttpError, requests.RequestException),
    )
    @rate_limit(yt_config.RECOMMENDED_DELAY_SECONDS)
    def search_videos(
        self, query: str, search_type: str, page_token: str = None
    ) -> Optional[Dict]:
        """
        Search for videos using YouTube Data API.

        Args:
            query: Search query string
            search_type: Search type (video, channel, playlist)
            page_token: Token for pagination

        Returns:
            Search response dictionary or None if failed
        """

        if not self.api_key:
            self.api_key = get_api_key()
        if not self.youtube:
            self.youtube = build(
                yt_config.YOUTUBE_API_SERVICE_NAME,
                yt_config.YOUTUBE_API_VERSION,
                developerKey=self.api_key,
            )

        search_params = {
            "q": query,
            "part": yt_config.SEARCH_PART,
            "type": search_type,
            "maxResults": yt_config.MAX_RESULTS_PER_SEARCH,
            "order": "date",
            "videoCaption": "closedCaption",
        }

        # Add optional filters
        if yt_config.PUBLISHED_AFTER and search_type == "video":
            search_params["publishedAfter"] = yt_config.PUBLISHED_AFTER
        if yt_config.VIDEO_CATEGORY_ID:
            search_params["videoCategoryId"] = yt_config.VIDEO_CATEGORY_ID

        if page_token:
            search_params["pageToken"] = page_token

        request = self.youtube.search().list(**search_params)  # type: ignore[union-attr]
        response = request.execute()

        logger.debug(
            "Search completed for '%s': %s results",
            query,
            len(response.get("items", [])),
        )

        return response

    @retry(
        max_attempts=yt_config.MAX_RETRIES,
        backoff=yt_config.RETRY_BACKOFF_FACTOR,
        exceptions=(HttpError, requests.RequestException),
    )
    @rate_limit(yt_config.RECOMMENDED_DELAY_SECONDS)
    def get_video_details(self, video_ids: str) -> Optional[Dict]:
        """
        Get detailed information for videos.
        """

        try:
            request = self.youtube.videos().list(  # type: ignore[union-attr]
                part=yt_config.VIDEO_PART, id=video_ids
            )
            response = request.execute()

            logger.debug(
                "Retrieved details for %s videos",
                len(response.get("items", [])),
            )

            return response

        except HttpError as e:
            if e.resp.status == 403:
                logger.error("Quota exceeded or API key invalid: %s", e)
            else:
                logger.error("HTTP error getting video details: %s", e)
            return None
        except Exception as e:
            logger.error("Error getting video details: %s", e)
            return None

    def _extract_video_ids_from_search(self, search_response: Dict) -> List[str]:
        """Extract video IDs from search API response."""
        return [
            item["id"]["videoId"]
            for item in search_response.get("items", [])
            if item.get("id", {}).get("kind") == "youtube#video"
        ]

    def _process_one_page(
        self,
        search_term: str,
        search_type: str,
        _page: int,
        next_page_token: Optional[str],
        all_videos: List[Dict],
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Run one page of search + details, append to all_videos.
        Returns (search_response, next_page_token) or (None, None) on failure.
        """
        search_response = self.search_videos(
            search_term, search_type, page_token=next_page_token
        )
        if not search_response:
            logger.warning("Failed to search for '%s'", search_term)
            return None, None

        video_ids = self._extract_video_ids_from_search(search_response)
        if not video_ids:
            return search_response, search_response.get("nextPageToken")

        new_video_ids = [vid for vid in video_ids if vid not in self.visited_videos]
        if not new_video_ids:
            return search_response, search_response.get("nextPageToken")

        video_details_response = self.get_video_details(",".join(new_video_ids))
        if not video_details_response:
            logger.warning("Failed to get video details for '%s'", search_term)
            return None, None

        for video_data in video_details_response.get("items", []):
            video_id = video_data.get("id")
            if not self._should_include_video(video_data):
                continue
            formatted_video = format_video_data(video_data)
            formatted_video["search_term"] = search_term
            all_videos.append(formatted_video)
            self.visited_videos.add(video_id)
            self._save_video_metadata(video_id, formatted_video)

        return search_response, search_response.get("nextPageToken")

    def process_search_term(self, search_term: str, search_type: str) -> List[Dict]:
        """Process a single search term and collect all videos."""
        logger.info("Processing search term: '%s'", search_term)

        all_videos = []
        next_page_token = None
        page = 1
        max_pages = 100

        while page <= max_pages:
            search_response, next_page_token = self._process_one_page(
                search_term, search_type, page, next_page_token, all_videos
            )
            if search_response is None:
                break

            logger.info("Page %s: collected %s videos", page, len(all_videos))

            if not next_page_token:
                break
            page += 1

        if page > max_pages:
            logger.info("Reached page limit for '%s'", search_term)

        logger.info("Completed '%s': %s new videos found", search_term, len(all_videos))
        return all_videos

    def _should_include_video(self, video_data: Dict) -> bool:
        """
        Check if video should be included based on filters.

        Args:
            video_data: Video data dictionary

        Returns:
            True if video should be included
        """
        content_details = video_data.get("contentDetails", {})

        # Check duration
        duration_iso = content_details.get("duration", "PT0S")
        duration_seconds = parse_duration(duration_iso)

        if duration_seconds < yt_config.MIN_DURATION_SECONDS:
            return False

        if (self.need_video and
            yt_config.MAX_DURATION_SECONDS
            and duration_seconds > yt_config.MAX_DURATION_SECONDS
        ):
            return False

        return True

    @retry(
        max_attempts=yt_config.MAX_RETRIES,
        backoff=yt_config.RETRY_BACKOFF_FACTOR,
        exceptions=(Exception,),
    )
    def _content_download(self, video_id: str) -> bool:
        """
        Download video and/or transcript using yt-dlp.

        Args:
            video_id: YouTube video ID

        Returns:
            True if download completed successfully.
        """
        if not self.ydl_opts:
            return False

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([video_url])

        # Move subtitle files to transcript directory if needed
        if self.need_video:
            self._move_subtitles_to_transcript_dir(video_id)

        logger.debug("Downloaded video/transcript for %s", video_id)

        return True

    def _move_subtitles_to_transcript_dir(self, video_id: str):
        """
        Move subtitle files to transcript directory.
        """
        transcript_dir = Path(yt_config.TRANSCRIPT_DIR)
        transcript_dir.mkdir(parents=True, exist_ok=True)

        video_dir = Path(yt_config.VIDEO_DOWNLOAD_DIR)

        subtitle_file = next(video_dir.glob(f"{video_id}*.vtt"), None)
        if subtitle_file:
            subtitle_file.rename(transcript_dir / subtitle_file.name)
            logger.debug(
                "Moved subtitle %s to transcript directory", subtitle_file.name
            )

    def _build_default_search_terms(self) -> List[str]:
        """Build default list of search terms (deduplicated)."""
        return list(
            set(
                C_PLUS_PLUS_CREATORS
                + C_PLUS_PLUS_CHANNELS
                + C_PLUS_PLUS_PLAYLISTS
                + SEARCH_TERMS
            )
        )

    def _search_type_for_term(self, search_term: str) -> str:
        """Return API search type for term: channel, playlist, or video."""
        if search_term in C_PLUS_PLUS_CHANNELS:
            return "channel"
        if search_term in C_PLUS_PLUS_PLAYLISTS:
            return "playlist"
        return "video"

    def scrape_all(self, search_terms: Optional[List[str]] = None) -> dict:
        """
        Scrape videos for all search terms.

        Args:
            search_terms: List of search terms (defaults to all from search_terms.py)

        Returns:
            Dictionary with scraping statistics
        """
        if search_terms is None:
            search_terms = self._build_default_search_terms()

        logger.info("Starting to scrape %s search terms", len(search_terms))

        successful_searches = 0
        start_time = time.time()
        search_results = []

        for i, search_term in enumerate(search_terms, 1):
            if search_term in self.progress.get("scraped_search_terms", []):
                logger.info(
                    "[%s/%s] Skipping already processed: '%s'",
                    i,
                    len(search_terms),
                    search_term,
                )
                continue

            search_type = self._search_type_for_term(search_term)
            logger.info("[%s/%s] Processing: '%s'", i, len(search_terms), search_term)

            try:
                search_results.extend(
                    self.process_search_term(search_term, search_type)
                )
                self.progress["scraped_search_terms"].append(search_term)
                save_progress(yt_config.PROGRESS_FILE, self.progress)
                logger.info("Progress saved: %s videos", len(search_results))
                successful_searches += 1
            except Exception as e:
                logger.error("Error searching for '%s': %s", search_term, e)
                break

        if not self.need_video:
            self.content_process(video_infos=search_results)

        elapsed = time.time() - start_time
        stats = {
            "total_search_terms": len(search_terms),
            "successful_searches": successful_searches,
            "total_videos": len(search_results),
            "elapsed_time": format_duration(elapsed),
        }
        logger.info("Scraping completed: %s", stats)
        return stats

    def find_missing_videos_from_json(self) -> List[str]:
        """
        Find video IDs that have metadata JSON but are not in visited set.

        Returns:
            List of video_id strings.
        """
        json_dir = Path(yt_config.METADATA_JSON_DIR)
        if not json_dir.exists():
            logger.warning("Metadata directory does not exist: %s", json_dir)
            return []

        json_files = list(json_dir.glob("*.json"))
        missing_ids = [f.stem for f in json_files if f.stem not in self.visited_videos]

        logger.info(
            "Scanning %s metadata files for missing videos...",
            len(json_files),
        )
        logger.info("Found %s videos without contents", len(missing_ids))
        return missing_ids

    def _normalize_video_info_item(self, item: Any) -> str:
        """Return video_id from item (dict with video_id key or raw string)."""
        if isinstance(item, dict):
            return item.get("video_id", "")
        return str(item)

    def content_process(self, video_infos: Optional[List[Any]] = None) -> dict:
        """
        Retry downloading videos that have metadata but missing content.

        Args:
            video_infos: List of video dicts or video_id strings; if None, uses find_missing_videos_from_json.

        Returns:
            Dictionary with retry statistics.
        """
        if video_infos is None:
            video_infos = self.find_missing_videos_from_json()

        if not video_infos:
            logger.info("No missing videos found. All videos are downloaded.")
            return {"total_missing": 0, "successful": 0, "failed": 0}

        logger.info("Retrying download for %s missing videos", len(video_infos))

        successful = 0
        start_time = time.time()

        for i, video_info in enumerate(video_infos, 1):
            video_id = self._normalize_video_info_item(video_info)
            if not video_id:
                continue

            logger.info("[%s/%s] downloading: %s", i, len(video_infos), video_id)

            try:
                if self._content_download(video_id):
                    successful += 1
                    logger.info("Successfully downloaded video %s", video_id)
                else:
                    logger.warning("No content for %s", video_id)
            except Exception as e:
                logger.error("Download error for video %s: %s", video_id, e)

        elapsed = time.time() - start_time
        stats = {
            "total_missing": len(video_infos),
            "successful": successful,
            "elapsed_time": format_duration(elapsed),
        }
        logger.info("Retry download completed: %s", stats)
        return stats


def main():
    """Main entry point for the scraper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape YouTube C++ videos by search term."
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of search terms to process"
    )
    parser.add_argument(
        "--retry-downloads",
        action="store_true",
        help="Only retry downloads for videos that have metadata JSON but missing transcript/video",
    )
    parser.add_argument(
        "--need-video",
        default=False,
        action="store_true",
        help="Download videos",
    )
    args = parser.parse_args()

    scraper = YouTubeCppScraper(need_video=args.need_video)

    if args.retry_downloads:
        stats = scraper.content_process()
        logger.info("Retry download statistics: %s", stats)
    else:
        stats = scraper.scrape_all()
        logger.info("Final statistics: %s", stats)


if __name__ == "__main__":
    main()
