"""
GitHub PR preprocessor for Pinecone RAG.

Reads PR JSON files under data/github/**/prs/*.json and builds one Document per PR.
Uses pr_to_md.convert_pr_to_markdown for content; content is built without any datetime.
Expected JSON shape:
- pr_info: pull request metadata from GitHub API
- comments: issue comments and/or review comments
- reviews: PR review objects
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config import GitConfig
from preprocessor.pr_to_md import convert_pr_to_markdown
from preprocessor.utility import get_timestamp_from_date, validate_content_length

logger = logging.getLogger(__name__)


def _get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get nested key from dict; return default when key path is missing."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _load_pr_document(json_path: Path, min_content_length: int) -> Optional[Document]:
    """Load one PR JSON file and convert it into a Document."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Skip %s: %s", json_path.name, exc)
        return None

    pr_info = data.get("pr_info")
    if not isinstance(pr_info, dict):
        logger.debug("Skip %s: missing pr_info", json_path.name)
        return None

    comments = data.get("comments") or []
    if not isinstance(comments, list):
        comments = []

    reviews = data.get("reviews") or []
    if not isinstance(reviews, list):
        reviews = []

    # Use pr_to_md for full markdown (reviews, comment tree, diff hunks);
    # content has no datetime and no pr_info url/title/state
    content = convert_pr_to_markdown(
        {"pr_info": pr_info, "comments": comments, "reviews": reviews},
        include_datetime=False,
        include_pr_title=False,
        include_pr_state=False,
        include_pr_url=False,
    )
    if not validate_content_length(content, min_length=min_content_length):
        logger.debug("Skip %s: content too short", json_path.name)
        return None

    number = pr_info.get("number", -1)
    metadata: Dict[str, Any] = {
        "type": "pr",
        "number": number,
        "title": (pr_info.get("title") or "").strip(),
        "url": pr_info.get("html_url", pr_info.get("url", "")) or "",
        "author": _get_nested(pr_info, "user", "login", default="") or "",
        "state": pr_info.get("state", "") or "",
        "created_at": get_timestamp_from_date(pr_info.get("created_at")),
        "updated_at": get_timestamp_from_date(pr_info.get("updated_at")),
        "closed_at": get_timestamp_from_date(pr_info.get("closed_at")),
    }

    return Document(page_content=content, metadata=metadata)


class GitPrPreprocessor:
    """Load GitHub PR JSON files from pr folders and produce Documents."""

    def __init__(self, config: Optional[GitConfig] = None):
        self.config = config or GitConfig()
        self.data_dir = Path(self.config.data_dir) / "pr"
        self.min_content_length = self.config.min_content_length

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        """Load PR JSON files from data/github/**/pr/*.json."""
        if not self.data_dir.exists():
            logger.warning("Git data dir does not exist: %s", self.data_dir)
            return []

        json_paths = sorted(self.data_dir.rglob("*.json"))
        if limit is not None:
            json_paths = json_paths[:limit]

        documents: List[Document] = []
        for json_path in json_paths:
            doc = _load_pr_document(json_path, self.min_content_length)
            if doc is not None:
                documents.append(doc)

        logger.info(
            "Loaded %d GitHub PR documents from %s",
            len(documents),
            self.data_dir,
        )
        return documents
