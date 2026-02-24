"""
Bugzilla issue preprocessor for Pinecone RAG.

Reads JSON files under data/bugs/** and builds one Document per bug.
Expected JSON shape:
- bug: bug metadata
- comments: list of comment objects
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from preprocessor.utility import get_timestamp_from_date

logger = logging.getLogger(__name__)


def _is_valid_content(text: str, min_length: int) -> bool:
    """Return True if text is non-empty and has at least min_length characters (after strip)."""
    return bool(text and len(text.strip()) >= min_length)


def _build_content(bug: Dict[str, Any], comments: List[Any]) -> str:
    """Build plain-text document content from bug metadata and comments."""
    lines: List[str] = []
    lines.append(f"Product: {bug.get('product', '')}")
    lines.append(f"Component: {bug.get('component', '')}")
    lines.append(f"Status: {bug.get('status', '')}")
    lines.append(f"Resolution: {bug.get('resolution', '')}")
    lines.append(f"Severity: {bug.get('severity', '')}")
    lines.append(f"Priority: {bug.get('priority', '')}")

    keywords = bug.get("keywords") or []
    if isinstance(keywords, list) and keywords:
        lines.append(f"Keywords: {', '.join(str(k) for k in keywords)}")

    lines.append("")
    lines.append("Description and comments:")

    for comment in comments:
        if not isinstance(comment, dict):
            continue
        text = (comment.get("text") or "").strip()
        if not text:
            continue
        creator = comment.get("creator") or ""
        created = comment.get("creation_time") or comment.get("time") or ""
        lines.append(f"\n--- Comment by {creator} ({created}) ---\n{text}")

    return "\n".join(lines).strip()


def _load_bug_document(json_path: Path, min_content_length: int) -> Optional[Document]:
    """Load one Bugzilla JSON file and convert it into a Document."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Skip %s: %s", json_path.name, exc)
        return None

    bug = data.get("bug")
    if not isinstance(bug, dict):
        logger.debug("Skip %s: missing bug object", json_path.name)
        return None

    comments = data.get("comments") or []
    if not isinstance(comments, list):
        comments = []

    content = _build_content(bug, comments)
    if not _is_valid_content(content, min_content_length):
        logger.debug("Skip %s: content too short", json_path.name)
        return None

    bug_id = bug.get("id", data.get("id", -1))
    is_open = bool(bug.get("is_open", False))
    last_change = bug.get("last_change_time")
    created = bug.get("creation_time")
    bug_url = bug.get("url") or f"https://bugs.llvm.org/show_bug.cgi?id={bug_id}"

    metadata: Dict[str, Any] = {
        "type": "issue-bugzilla",
        "number": bug_id,
        "title": (bug.get("summary") or "").strip(),
        "url": bug_url,
        "author": bug.get("creator", "") or "",
        "state": bug.get("status", "") or "",
        "state_reason": bug.get("resolution", "") or "",
        "created_at": get_timestamp_from_date(created or "", 0.0),
        "updated_at": get_timestamp_from_date(last_change or "", 0.0),
        "closed_at": 0.0 if is_open else get_timestamp_from_date(last_change or "", 0.0),
    }

    return Document(page_content=content, metadata=metadata)


class BugIssuePreprocessor:
    """Load Bugzilla issue JSON files from data/bugs and produce Documents."""

    def __init__(
        self, data_dir: str = "data/github/Clang/bugs", min_content_length: int = 10
    ):
        """Initialize with the directory containing Bugzilla JSON files and minimum content length."""
        self.data_dir = Path(data_dir)
        self.min_content_length = min_content_length

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        """Load Bugzilla JSON files from data/github/Clang/bugs/**/*.json."""
        if not self.data_dir.exists():
            logger.warning("Bug data dir does not exist: %s", self.data_dir)
            return []

        json_paths = sorted(self.data_dir.rglob("*.json"))
        if limit is not None:
            json_paths = json_paths[:limit]

        documents: List[Document] = []
        for json_path in json_paths:
            if json_path.name.startswith("."):
                continue
            doc = _load_bug_document(json_path, self.min_content_length)
            if doc is not None:
                documents.append(doc)

        logger.info(
            "Loaded %d Bugzilla issue documents from %s", len(documents), self.data_dir
        )
        return documents
