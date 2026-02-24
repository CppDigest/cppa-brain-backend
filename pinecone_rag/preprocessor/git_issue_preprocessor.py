"""
GitHub issue/PR preprocessor for Pinecone RAG.

Loads GitHub issue or PR JSON files from data/github (e.g. data/github/Clang/issue/84062.json),
extracts issue_info or pr_info and comments, and produces LangChain Documents with metadata:
doc_id, title, url, author, timestamp, type (github-issue | github-pr), repository, number, state, labels.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config import GitConfig
from preprocessor.utility import get_timestamp_from_date, validate_content_length

logger = logging.getLogger(__name__)


def _get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get nested key from dict; return default if any key is missing."""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
    return data


def _extract_labels(info: Dict[str, Any]) -> List[str]:
    """Extract label names from issue/PR info."""
    labels_raw = info.get("labels") or []
    return [
        lb.get("name") if isinstance(lb, dict) else str(lb) for lb in labels_raw if lb
    ]


def _build_content_parts(
    labels: List[str], body: str, comments: List[Any]
) -> List[str]:
    """Build list of content segments: title, body, then each comment with header."""
    parts = [f"Labels: {', '.join(labels)}\n\n", body] if labels else [body]
    for com in comments:
        com_body = (com.get("body") or "").strip()
        if not com_body or com_body == "":
            continue
        com_user = _get_nested(com, "user", "login", default="") or ""
        com_created = com.get("created_at") or ""
        parts.append(f"\n\n--- Comment by {com_user} ({com_created}) ---\n\n{com_body}")
    return parts


def _info_to_document(
    info: Dict[str, Any],
    comments: List[Any],
    json_path: Path,
    min_content_length: int,
) -> Optional[Document]:
    """
    Build one Document from GitHub issue or PR info dict and comments list.

    Shared by issue and PR; doc_type is "github-issue" or "github-pr".
    """
    html_url = info.get("html_url") or info.get("url") or ""
    if not html_url:
        return None

    title = (info.get("title") or "").strip()
    body = (info.get("body") or "").strip()
    author = _get_nested(info, "user", "login", default="") or ""
    created_at = get_timestamp_from_date(info.get("created_at"))
    updated_at = get_timestamp_from_date(info.get("updated_at"))
    closed_at = get_timestamp_from_date(
        info.get("closed_at", info.get("pushed_at", ""))
    )
    labels = _extract_labels(info)
    state = info.get("state", "") or ""
    state_reason = info.get("state_reason", "") or ""

    number = info.get("number", -1)

    content = "".join(_build_content_parts(labels, body, comments)).strip()
    if not validate_content_length(content, min_length=min_content_length):
        logger.debug("Skip %s: content too short", json_path.name)
        return None

    meta: Dict[str, Any] = {
        "author": author or "",
        "title": title or "",
        "number": number or -1,
        "url": html_url or "",
        "created_at": created_at or 0.0,
        "updated_at": updated_at or 0.0,
        "closed_at": closed_at or 0.0,
        "type": "issue",
        "state": state or "",
        "state_reason": state_reason or "",
    }

    return Document(page_content=content, metadata=meta)


def _issue_json_to_document(
    json_path: Path,
    data: Dict[str, Any],
    min_content_length: int,
) -> Optional[Document]:
    """Build one Document from a GitHub issue JSON file (issue_info + comments)."""
    info = data.get("issue_info")
    if not info or not isinstance(info, dict):
        logger.debug("Skip %s: no issue_info", json_path.name)
        return None
    comments = data.get("comments") or []
    return _info_to_document(info, comments, json_path, min_content_length)


def _pr_json_to_document(
    json_path: Path,
    data: Dict[str, Any],
    min_content_length: int,
) -> Optional[Document]:
    """Build one Document from a GitHub PR JSON file (pr_info + comments)."""
    info = data.get("pr_info")
    if not info or not isinstance(info, dict):
        logger.debug("Skip %s: no pr_info", json_path.name)
        return None
    comments = data.get("comments") or []
    if isinstance(comments, dict):
        comments = []
    return _info_to_document(info, comments, json_path, min_content_length, "github-pr")


def _load_one_json(
    json_path: Path,
    min_content_length: int,
) -> Optional[Document]:
    """Load one JSON file and return an issue or PR document, or None."""
    try:
        path_str = str(json_path)
        path_str = path_str.lower()
        raw = json_path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Skip %s: %s", json_path.name, e)
        return None

    if "issue_info" in data:
        return _issue_json_to_document(json_path, data, min_content_length)
    if "pr_info" in data:
        return _pr_json_to_document(json_path, data, min_content_length)
    logger.debug("Skip %s: no issue_info or pr_info", json_path.name)
    return None


class GitIssuePreprocessor:
    """
    Process GitHub issue/PR JSON files under data/github for RAG.

    Discovers all *.json under data_dir (e.g. data/github/Clang/issue/*.json),
    parses issue_info or pr_info + comments, and produces one Document per file.
    """

    def __init__(self, config: Optional[GitConfig] = None):
        self.config = config or GitConfig()
        self.data_dir = Path(self.config.data_dir) / "issue"
        self.min_content_length = self.config.min_content_length

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        """
        Load all GitHub issue/PR JSON files and convert to Documents.

        Returns one Document per JSON file (issue or PR with comments).
        """
        if not self.data_dir.exists():
            logger.warning("Git data dir does not exist: %s", self.data_dir)
            return []

        json_paths = sorted(self.data_dir.rglob("*.json"))
        if limit is not None:
            json_paths = json_paths[:limit]

        documents: List[Document] = []
        for json_path in json_paths:
            doc = _load_one_json(json_path, self.min_content_length)
            if doc is not None:
                documents.append(doc)

        logger.info(
            "Loaded %d GitHub issue/PR documents from %s",
            len(documents),
            self.data_dir,
        )
        return documents
