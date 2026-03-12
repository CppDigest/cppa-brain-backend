"""
Phabricator PR-like preprocessor for Pinecone RAG.

Reads markdown files under data/phabricator/** and builds one Document per file.
Expected markdown header:
- # D<number> <title> [Open|Closed]
- > Username: <author>
- > Created at: <date text>
- > Url: https://reviews.llvm.org/D<number>
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(
    r"^#\s*D(?P<number>\d+)\s+(?P<title>.+?)\s+\[(?P<state>[^\]]+)\]\s*$"
)
_USERNAME_RE = re.compile(r"^>\s*Username:\s*(.+?)\s*$", re.MULTILINE)
_CREATED_AT_RE = re.compile(r"^>\s*Created at:\s*(.+?)\s*$", re.MULTILINE)
_URL_RE = re.compile(r"^>\s*Url:\s*(https?://\S+)\s*$", re.MULTILINE)

_CLOSED_STATES = {"closed", "abandoned", "merged"}


def _is_valid_content(text: str, min_length: int) -> bool:
    """Return True if text is non-empty and has at least min_length characters (after strip)."""
    return bool(text and len(text.strip()) >= min_length)


def _parse_created_at_to_timestamp(value: str) -> float:
    """Parse Phabricator 'Created at' date string to Unix timestamp (UTC). Returns 0.0 on empty or parse failure."""
    if not value:
        return 0.0

    patterns = [
        "%b %d %Y, %I:%M %p",  # Jan 18 2023, 5:56 PM
        "%b %d %Y, %H:%M",  # Jan 18 2023, 17:56
    ]
    for pattern in patterns:
        try:
            dt = datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            logger.debug("Date parse failed for pattern '%s': %s", pattern, value)
            continue
    return 0.0


def _extract_metadata(md_text: str, file_path: Path) -> Dict[str, Any]:
    """Parse markdown content for D-number, title, state, author, URL and timestamps; return metadata dict."""
    lines = md_text.splitlines()
    first_line = lines[0].strip() if lines else ""

    header_match = _HEADER_RE.match(first_line)
    if header_match:
        number = int(header_match.group("number"))
        title = header_match.group("title").strip()
        state = header_match.group("state").strip()
    else:
        number = -1
        title = file_path.stem
        state = ""

    user_match = _USERNAME_RE.search(md_text)
    url_match = _URL_RE.search(md_text)

    author = user_match.group(1).strip() if user_match else ""
    url = url_match.group(1).strip() if url_match else ""

    if not url and number > 0:
        url = f"https://reviews.llvm.org/D{number}"

    # Collect all "Created at:" timestamps from PR header + all comments/reviews.
    # The first match is the PR's own creation time; the maximum is the last activity.
    all_timestamps = [
        _parse_created_at_to_timestamp(raw.strip())
        for raw in _CREATED_AT_RE.findall(md_text)
    ]
    valid_timestamps = [ts for ts in all_timestamps if ts > 0.0]

    created_at = min(valid_timestamps) if valid_timestamps else 0.0
    last_activity = max(valid_timestamps) if valid_timestamps else 0.0
    updated_at = last_activity
    closed_at = last_activity if state.lower() in _CLOSED_STATES else 0.0

    return {
        "type": "pr-phabricator",
        "number": number,
        "title": title,
        "url": url,
        "author": author,
        "state": state.lower(),
        "state_reason": "",
        "created_at": created_at,
        "updated_at": updated_at,
        "closed_at": closed_at,
    }


def _load_pr_document(md_path: Path, min_content_length: int) -> Optional[Document]:
    """Load one Phabricator markdown file and convert it into a Document with extracted metadata."""
    try:
        content = md_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.debug("Skip %s: %s", md_path.name, exc)
        return None

    if not _is_valid_content(content, min_content_length):
        logger.debug("Skip %s: content too short", md_path.name)
        return None

    metadata = _extract_metadata(content, md_path)
    return Document(page_content=content, metadata=metadata)


class PhabricatorPrPreprocessor:
    """
    Load Phabricator PR-style markdown files and produce LangChain Documents.

    Expects markdown with header lines for D-number, title, state, username,
    created-at, and URL. Call load_documents() to scan the data directory and
    return a list of Document instances with extracted metadata.
    """

    def __init__(
        self,
        data_dir: str = "data/github/Clang/phabricator",
        min_content_length: int = 10,
    ):
        """Initialize with the directory containing Phabricator markdown files and minimum content length."""
        self.data_dir = Path(data_dir)
        self.min_content_length = min_content_length

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        """Load Phabricator markdown files from data/github/Clang/phabricator/**/*.md."""
        if not self.data_dir.exists():
            logger.warning("Phabricator data dir does not exist: %s", self.data_dir)
            return []

        md_paths = sorted(self.data_dir.rglob("*.md"))
        if limit is not None:
            md_paths = md_paths[:limit]

        documents: List[Document] = []
        for md_path in md_paths:
            doc = _load_pr_document(md_path, self.min_content_length)
            if doc is not None:
                documents.append(doc)

        logger.info(
            "Loaded %d Phabricator PR documents from %s",
            len(documents),
            self.data_dir,
        )
        return documents
