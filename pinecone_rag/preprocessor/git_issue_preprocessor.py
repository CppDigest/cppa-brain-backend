"""
GitHub issue preprocessor for Pinecone RAG (LLVM only).

Loads GitHub issue JSON files from ``data/github/**/issue/*.json`` (e.g.
``data/github/Clang/issue/84062.json``), extracts ``issue_info`` and comments,
and produces LangChain Documents for chunking and embedding. Only issues from the
LLVM GitHub organization (e.g. llvm/llvm-project) are included; others are skipped.

Each Document has metadata: title, url, author, number, state, state_reason,
created_at, updated_at, closed_at, and type ``"issue"``. Content is built from
labels, body, and comment threads.
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
    """Get a nested key from a dict by path; return default if any key is missing.

    Args:
        data: Root dictionary to traverse.
        *keys: Sequence of keys to follow (e.g. "user", "login").
        default: Value to return if the path is missing or a step is not a dict.

    Returns:
        The value at the key path, or default.
    """
    for key in keys:
        if not isinstance(data, dict):
            return default
        if key not in data:
            return default
        data = data[key]
    return data


def _extract_labels(info: Dict[str, Any]) -> List[str]:
    """Extract label names from issue info.

    Args:
        info: GitHub issue object containing a "labels" array.

    Returns:
        List of label name strings (from each label's "name" field or str(label)).
    """
    labels_raw = info.get("labels") or []
    labels: List[str] = []
    for lb in labels_raw:
        if not lb:
            continue
        if isinstance(lb, dict):
            name = (lb.get("name") or "").strip()
            if name:
                labels.append(name)
        else:
            labels.append(str(lb))
    return labels


def _build_content_parts(
    labels: List[str], body: str, comments: List[Any]
) -> List[str]:
    """Build list of content segments for the Document page_content.

    Args:
        labels: Label names to prefix (optional "Labels: ..." line).
        body: Issue body text.
        comments: List of comment dicts with body, user.login, created_at.

    Returns:
        List of strings to be joined: labels line (if any), body, then each
        comment with a "--- Comment by {user} ({date}) ---" header.
    """
    parts = [f"Labels: {', '.join(labels)}\n\n", body] if labels else [body]
    for com in comments:
        com_body = (com.get("body") or "").strip()
        if not com_body:
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
    """Build one Document from GitHub issue info and comments.

    Args:
        info: GitHub issue object (title, body, user, state, labels, etc.).
        comments: List of comment dicts (body, user, created_at).
        json_path: Path to the source JSON file (used for logging).
        min_content_length: Minimum character length for content; shorter skips.

    Returns:
        A Document with combined content (labels, body, comments) and metadata,
        or None if html_url is missing or content is too short.
    """
    html_url = info.get("html_url", "").strip()
    if not html_url:
        return None

    title = (info.get("title") or "").strip()
    body = (info.get("body") or "").strip()
    author = _get_nested(info, "user", "login", default="") or ""
    created_at = get_timestamp_from_date(info.get("created_at"))
    updated_at = get_timestamp_from_date(info.get("updated_at"))
    closed_at = info.get("closed_at")
    if closed_at:
        closed_at = get_timestamp_from_date(closed_at)
    else:
        closed_at = 0.0
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
    """Build one Document from a GitHub issue JSON file (issue_info + comments).

    Args:
        json_path: Path to the JSON file (for logging).
        data: Parsed JSON root; must contain "issue_info" and optionally "comments".
        min_content_length: Minimum content length; shorter documents are skipped.

    Returns:
        A Document with type "issue", or None if issue_info is missing or invalid.
    """
    info = data.get("issue_info")
    if not info or not isinstance(info, dict):
        logger.debug("Skip %s: no issue_info", json_path.name)
        return None
    comments = data.get("comments") or []
    return _info_to_document(info, comments, json_path, min_content_length)


def _load_one_json(
    json_path: Path,
    min_content_length: int,
) -> Optional[Document]:
    """Load one JSON file and return an issue Document, or None.

    Args:
        json_path: Path to the JSON file.
        min_content_length: Minimum content length; shorter documents are skipped.

    Returns:
        A Document for the issue, or None if the file is invalid, missing
        issue_info, or content is too short.
    """
    try:
        raw = json_path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Skip %s: %s", json_path.name, e)
        return None

    if not isinstance(data, dict):
        logger.debug("Skip %s: JSON root is not an object", json_path.name)
        return None
    if "issue_info" not in data:
        logger.debug("Skip %s: no issue_info", json_path.name)
        return None
    return _issue_json_to_document(json_path, data, min_content_length)


class GitIssuePreprocessor:
    """Load LLVM GitHub issue JSON files and produce Documents.

    Discovers all ``*.json`` under ``config.data_dir / "issue"`` (e.g.
    ``data/github/Clang/issue/*.json``), parses each as an issue with
    issue_info and comments. Only issues from the LLVM GitHub organization
    (e.g. llvm/llvm-project) are included; others are skipped. Returns a list
    of LangChain Documents for RAG ingestion.
    """

    def __init__(self, config: Optional[GitConfig] = None):
        """Initialize the preprocessor with optional GitConfig.

        Args:
            config: Git configuration (data_dir, min_content_length). If None,
                uses default GitConfig().
        """
        self.config = config or GitConfig()
        self.data_dir = Path(self.config.data_dir) / "issue"
        self.min_content_length = self.config.min_content_length

    def load_documents(self, limit: Optional[int] = None) -> List[Document]:
        """Load issue JSON files from the configured issue directory and convert to Documents.

        Args:
            limit: Optional maximum number of JSON files to process (by sorted path).
                If None, all discovered *.json files are processed.

        Returns:
            List of Documents (one per valid issue JSON file) with combined
            content (labels, body, comments) and metadata. Skips invalid files
            and those with content below min_content_length.
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
            "Loaded %d GitHub issue documents from %s",
            len(documents),
            self.data_dir,
        )
        return documents
