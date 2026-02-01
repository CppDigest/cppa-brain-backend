"""
Utility functions for preprocessors.

Common functions shared across different preprocessor classes.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Common date formats used across different data sources
DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",  # ISO format without timezone
    "%Y-%m-%dT%H:%M:%SZ",  # ISO format with Z timezone
    "%Y-%m-%dT%H:%M:%S.%f",  # ISO format with microseconds
    "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds and Z
    "%Y-%m-%d %H:%M:%S",  # Standard format
    "%Y-%m-%d",  # Date only
    "%d/%m/%Y",  # European format
    "%m/%d/%Y",  # US format
    "%d-%m-%Y",  # European format with dashes
    "%m-%d-%Y",  # US format with dashes
    "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format (e.g., "Sun, 09 Oct 2011 20:30:18 +0200")
    "%a, %d %b %Y %H:%M:%S",  # RFC 2822 without timezone
    "%d %b %Y",  # "09 Oct 2011"
    "%b %d, %Y",  # "Oct 09, 2011"
    "%Y-%m-%d %H:%M:%S%z",  # ISO with timezone offset
]


def get_timestamp_from_date(date_str: str, default: Optional[float] = None) -> float:
    """
    Convert date string to Unix timestamp.

    Tries multiple date formats to parse the input string.

    Args:
        date_str: Date string in various formats
        default: Default timestamp to return if parsing fails (default: current time)

    Returns:
        Unix timestamp (float)

    Examples:
        >>> get_timestamp_from_date("2023-01-15T10:30:00Z")
        1673782200.0
        >>> get_timestamp_from_date("2023-01-15")
        1673740800.0
        >>> get_timestamp_from_date("")  # Returns current timestamp
    """
    if not date_str or not date_str.strip():
        return default if default is not None else datetime.now().timestamp()

    date_str = date_str.strip()

    # Normalize ISO strings with trailing timezone so DATE_FORMATS can parse them
    if "T" in date_str and (
        date_str.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", date_str)
    ):
        date_str = date_str.replace("Z", "").strip()
        date_str = re.sub(r"[+-]\d{2}:?\d{2}$", "", date_str).strip()

    # Try standard formats (DATE_FORMATS includes ISO with/without Z and microseconds)
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue

    # Try parsing date components manually (e.g., "2011-10-09")
    try:
        # Remove soft hyphens and other invisible characters
        clean_date = date_str.replace("\xad", "").replace("\u200b", "")
        if "-" in clean_date:
            elements = clean_date.split("-")
            if len(elements) >= 3:
                year = int(elements[0])
                month = int(elements[1])
                day = int(elements[2])
                dt = datetime(year, month, day)
                return dt.timestamp()
    except (ValueError, IndexError):
        pass

    # If all parsing fails, return default or current time
    logger.debug(f"Failed to parse date: {date_str}, using default")
    return default if default is not None else datetime.now().timestamp()


def parse_date_to_iso(date_str: str) -> Optional[str]:
    """
    Parse date string to ISO format (YYYY-MM-DDTHH:MM:SS).

    Args:
        date_str: Date string in various formats

    Returns:
        ISO format date string, or None if parsing fails

    Examples:
        >>> parse_date_to_iso("Sun, 09 Oct 2011 20:30:18 +0200")
        '2011-10-09T20:30:18'
        >>> parse_date_to_iso("2023-01-15")
        '2023-01-15T00:00:00'
    """
    if not date_str or not date_str.strip():
        return None

    timestamp = get_timestamp_from_date(date_str)
    dt = datetime.fromtimestamp(timestamp)
    return dt.isoformat()


def clean_text(text: str, remove_extra_spaces: bool = True) -> str:
    """
    Clean and normalize text content.

    Args:
        text: Input text to clean
        remove_extra_spaces: Whether to remove extra whitespace

    Returns:
        Cleaned text

    Examples:
        >>> clean_text("  Hello   world  ")
        'Hello world'
        >>> clean_text("Text\\n\\n\\nMore text")
        'Text\\n\\nMore text'
    """
    if not text:
        return ""

    # Remove soft hyphens and other invisible characters
    text = (
        text.replace("\xad", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )

    # Normalize line breaks
    text = re.sub(r"\r\n", "\n", text)  # Windows line breaks
    text = re.sub(r"\r", "\n", text)  # Old Mac line breaks

    if remove_extra_spaces:
        # Remove multiple spaces
        text = re.sub(r" +", " ", text)
        # Remove multiple newlines (keep max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove spaces at start/end of lines
        text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def extract_video_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract video ID from various filename formats.
    Examples:
        >>> extract_video_id_from_filename("VAMvr1rXmg8.en.vtt")
        'VAMvr1rXmg8'
        >>> extract_video_id_from_filename("VAMvr1rXmg8.json")
        'VAMvr1rXmg8'
    """
    if not filename:
        return None

    # Remove extension
    name = Path(filename).stem

    # If there's a language code (e.g., .en), remove it
    if "." in name:
        parts = name.split(".")
        # Video ID is typically the first part
        return parts[0] if parts else None

    return name


def validate_content_length(content: str, min_length: int = 50) -> bool:
    """
    Validate that content meets minimum length requirement.

    Args:
        content: Content string to validate
        min_length: Minimum required length (default: 50)

    Returns:
        True if content is valid, False otherwise
    """
    if not content:
        return False

    cleaned = content.strip()
    return len(cleaned) >= min_length


def normalize_metadata_value(value: Any) -> Any:
    """
    Normalize metadata values for consistent storage.

    Converts values to appropriate types and handles edge cases.

    Args:
        value: Metadata value to normalize

    Returns:
        Normalized value

    Examples:
        >>> normalize_metadata_value("123")
        123
        >>> normalize_metadata_value("")
        ''
        >>> normalize_metadata_value(None)
        ''
    """
    if value is None:
        return ""

    # Convert string numbers to integers if appropriate
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return ""
        # Try to convert to int if it's a numeric string
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)
        except (ValueError, AttributeError):
            pass

    # Convert lists to ensure they're proper lists
    if isinstance(value, (list, tuple)):
        return list(value)

    return value


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value

    Examples:
        >>> safe_int("123")
        123
        >>> safe_int("abc", default=0)
        0
        >>> safe_int(None, default=0)
        0
    """
    if value is None:
        return default

    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value
    """
    if value is None:
        return default

    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_url_from_text(text: str) -> Optional[str]:
    """
    Extract URL from text content.

    Args:
        text: Text that may contain URLs

    Returns:
        First URL found, or None
    """
    if not text:
        return None

    # Common URL patterns
    url_patterns = [
        r"https?://[^\s]+",  # http:// or https://
        r"www\.[^\s]+",  # www.example.com
    ]

    for pattern in url_patterns:
        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            # Clean up trailing punctuation
            url = url.rstrip(".,;:!?)")
            return url

    return None


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Text that may contain HTML tags

    Returns:
        Text with HTML tags removed
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")

    return text.strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def get_file_stats(directory: Path, pattern: str = "*") -> Dict[str, Any]:
    """
    Get statistics about files in a directory.

    Args:
        directory: Directory path to analyze
        pattern: File pattern to match (default: "*")

    Returns:
        Dictionary with file statistics
    """
    stats = {
        "total_files": 0,
        "total_size": 0,
        "file_extensions": {},
    }

    if not directory.exists():
        return stats

    for file_path in directory.rglob(pattern):
        if file_path.is_file():
            stats["total_files"] += 1
            try:
                stats["total_size"] += file_path.stat().st_size
                ext = file_path.suffix.lower()
                stats["file_extensions"][ext] = stats["file_extensions"].get(ext, 0) + 1
            except OSError:
                pass

    return stats


def time_to_seconds(time_str: str) -> float:
    """
    Convert time string (HH:MM:SS.mmm) to seconds.

    Supports VTT time format and other time formats.

    Args:
        time_str: Time string in format "HH:MM:SS.mmm" or "HH:MM:SS"

    Returns:
        Time in seconds as float

    Examples:
        >>> time_to_seconds("00:01:30.500")
        90.5
        >>> time_to_seconds("01:23:45.123")
        5025.123
        >>> time_to_seconds("00:00:05")
        5.0
    """
    try:
        parts = time_str.split(":")
        if len(parts) != 3:
            logger.warning(f"Invalid time format: {time_str}")
            return 0.0

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split(".")
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing time string {time_str}: {e}")
        return 0.0
