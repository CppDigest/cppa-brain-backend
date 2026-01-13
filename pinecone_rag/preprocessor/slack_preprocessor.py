"""
Slack preprocessing pipeline for LangChain RAG

Loads Slack chat data from PostgreSQL database and converts to documents.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.documents import Document
from tqdm import tqdm
import logging

from config import SlackConfig

try:
    from psycopg2.extras import RealDictCursor
    from psycopg2 import pool
except ImportError as e:
    RealDictCursor = None  # type: ignore[assignment]
    pool = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class SlackPreprocessor:
    """Process Slack chat data from PostgreSQL database for RAG"""

    def __init__(self, slack_config: Optional[SlackConfig] = None):
        """
        Initialize Slack preprocessor.

        Args:
            slack_config: SlackConfig instance (loads from env vars if not provided)
        """
        self._validate_imports()
        self.slack_config = slack_config or SlackConfig()
        self._connection_pool: Optional[pool.ThreadedConnectionPool] = None

        # Greeting and unessential words to filter
        self._greeting_words = {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "greetings",
            "howdy",
            "sup",
            "what's up",
            "yo",
            "hii",
            "helloo",
            "thanks",
            "thank you",
            "thx",
            "ty",
            "appreciate it",
            "cheers",
            "nice to meet you",
            "happy to be here",
            "happy to have you here",
            "glad to see you",
            "glad to see you here",
            "glad to be here",
            "bye",
            "goodbye",
            "see you later",
            "see you soon",
            "see you tomorrow",
            "see you next week",
            "see you next month",
            "see you next year",
            "see you in the future",
        }
        self._unessential_words = {
            "ok",
            "okay",
            "sure",
            "yeah",
            "yep",
            "yup",
            "nope",
            "nah",
            "lol",
            "haha",
            "hahaha",
            "hehe",
            "lmao",
            "rofl",
            "👍",
            "👎",
            "😊",
            "😄",
            "😀",
            "👍🏻",
            "👌",
            "got it",
            "gotcha",
            "nice",
            "awesome",
            "great",
            "uhm",
            "um",
            "uh",
            "erm",
            "of course",
        }

    def _validate_imports(self) -> None:
        """Validate that required imports are available."""
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Missing optional dependencies required for SlackPreprocessor. "
                "Install with: pip install psycopg2-binary"
            ) from _IMPORT_ERROR

    def _get_connection(self):
        """Get a database connection from the pool or create a new one."""
        if self._connection_pool is None:
            try:
                self._connection_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=5,
                    dsn=(
                        f"host={self.slack_config.db_host} port={self.slack_config.db_port} "
                        f"dbname={self.slack_config.db_name} user={self.slack_config.db_user} "
                        f"password={self.slack_config.db_password}"
                    ),
                )
                logger.info("Created PostgreSQL connection pool")
            except Exception as e:
                logger.error(f"Failed to create connection pool: {e}")
                raise ConnectionError(
                    f"Cannot connect to PostgreSQL database. Error: {e}"
                ) from e

        return self._connection_pool.getconn()

    def _return_connection(self, conn):
        """Return a connection to the pool."""
        if self._connection_pool:
            self._connection_pool.putconn(conn)

    def _close_pool(self):
        """Close the connection pool."""
        if self._connection_pool:
            self._connection_pool.closeall()
            self._connection_pool = None
            logger.info("Closed PostgreSQL connection pool")

    def load_messages(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Document]:
        """
        Load Slack messages from PostgreSQL database.

        Args:
            date_from: Optional start date for filtering messages
            date_to: Optional end date for filtering messages

        Returns:
            List of Document objects created from Slack messages
        """
        try:
            messages = self._fetch_messages(
                date_from=date_from,
                date_to=date_to,
            )
            grouped_messages = self.filter_and_group_messages(messages)
            documents = self._convert_messages_to_documents(grouped_messages)
            logger.info(f"Loaded {len(documents)} documents from Slack database")
            return documents
        except Exception as e:
            logger.error(f"Error loading Slack messages: {e}")
            raise
        finally:
            self._close_pool()

    def _fetch_messages(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch messages from PostgreSQL database."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query, params = self._build_query(date_from, date_to)
            cursor.execute(query, params)
            return [dict(msg) for msg in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching messages from database: {e}")
            raise
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)

    def _build_query(
        self, date_from: Optional[datetime], date_to: Optional[datetime]
    ) -> tuple[str, List[Any]]:
        """Build SQL query with conditions."""
        query = f"""SELECT sm.ts as id, sm.ts, sm.message, sm.thread_ts,
            sc.channel_id, st.team_id,
            su.display_name as display_name, su.username as user_name
            FROM {self.slack_config.message_table} as sm
            LEFT JOIN {self.slack_config.channel_table} as sc ON sm.channel_id = sc.id
            LEFT JOIN {self.slack_config.team_table} as st ON sc.team_id = st.id
            LEFT JOIN {self.slack_config.user_table} as su ON sm.user_id = su.email_id
        """
        conditions = []
        params = []

        if date_from:
            conditions.append("ts >= %s")
            params.append(date_from)
        if date_to:
            conditions.append("ts <= %s")
            params.append(date_to)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY ts ASC"

        return query, params

    def filter_and_group_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and group messages by thread, merging consecutive messages from same user."""
        if not messages:
            return []

        thread_groups = self._group_by_thread(messages)
        grouped_messages = []
        for thread_ts, thread_messages in thread_groups.items():
            thread_messages.sort(key=lambda m: float(m.get("ts", 0)))
            if thread_ts is not None:
                if group := self._merge_thread_messages(thread_messages, thread_ts):
                    grouped_messages.append(group)
            else:
                grouped_messages.extend(
                    self._merge_none_thread_messages(thread_messages)
                )
        return grouped_messages

    def _group_by_thread(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[Optional[str], List[Dict[str, Any]]]:
        """Group messages by thread_ts."""
        groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for msg in messages:
            thread_ts = msg.get("thread_ts")
            if thread_ts not in groups:
                groups[thread_ts] = []
            groups[thread_ts].append(msg)
        return groups

    def _merge_thread_messages(
        self, thread_messages: List[Dict[str, Any]], thread_ts: str
    ) -> Optional[Dict[str, Any]]:
        """Merge all messages in a thread into one text after filtering unessential words."""
        if not thread_messages:
            return None

        merged_parts, message_ids = self._extract_valid_messages(thread_messages)
        if not merged_parts:
            return None

        first_msg = thread_messages[0]
        return {
            "id": first_msg.get("id", ""),
            "message_ids": message_ids,
            "text": " ".join(merged_parts),
            "user_name": self._get_user_name(first_msg),
            "channel_id": first_msg.get("channel_id", ""),
            "ts": first_msg.get("ts"),
            "thread_ts": thread_ts,
            "is_grouped": True,
            "team_id": first_msg.get("team_id", ""),
        }

    def _extract_valid_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[str], List[str]]:
        """Extract and filter valid messages, returning text parts and message IDs."""
        merged_parts = []
        message_ids = []
        for msg in messages:
            text = msg.get("message", "").strip()
            if not text:
                continue
            filtered = self._filter_unessential_words(text)
            if filtered and len(filtered.strip()) >= 10:
                merged_parts.append(filtered)
                if msg_id := msg.get("id"):
                    message_ids.append(msg_id)
        return merged_parts, message_ids

    def _get_user_name(self, msg: Dict[str, Any]) -> str:
        """Extract user name from message."""
        return msg.get("display_name") or msg.get("user_name") or "unknown"

    def _merge_none_thread_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge consecutive messages from the same user."""
        if not messages:
            return []
        first_group = self._merge_by_user_name(messages)
        final_group = self._merge_consecutive_messages(first_group)
        return final_group

    def _merge_by_user_name(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge messages from the same user."""
        merged_groups = []
        current_group: Optional[Dict[str, Any]] = None

        for msg in messages:
            text = self._filter_unessential_words(msg.get("message", "").strip())
            if not text:
                continue

            user_name = self._get_user_name(msg)
            if (
                current_group is not None
                and current_group.get("user_name") == user_name
                and self._is_consecutive_message(current_group, msg)
            ):
                assert current_group is not None  # Type narrowing
                current_group["text"] += " " + text
                current_group["message_ids"].append(msg.get("id"))
                current_group["ts"] = msg.get("ts")
            else:
                if current_group is not None:
                    merged_groups.append(current_group)
                current_group = self._create_message_group(msg, user_name, text)

        if current_group is not None:
            merged_groups.append(current_group)
        return merged_groups

    def _create_message_group(
        self, msg: Dict[str, Any], user_name: str, text: str
    ) -> Dict[str, Any]:
        """Create a new message group dictionary."""
        return {
            "id": msg.get("id"),
            "message_ids": [msg.get("id")],
            "text": text,
            "user_name": user_name,
            "channel_id": msg.get("channel_id", ""),
            "ts": msg.get("ts"),
            "thread_ts": msg.get("thread_ts"),
            "is_grouped": True,
            "start_ts": msg.get("ts"),
            "team_id": msg.get("team_id", ""),
        }

    def _merge_consecutive_messages(
        self, groups: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge consecutive message groups that are close in time."""
        if not groups:
            return []
        merged_groups = []
        current_group = groups[0].copy()
        for group in groups[1:]:
            if self._is_consecutive_message(current_group, group):
                current_group["text"] += " " + group["text"]
                current_group["message_ids"].extend(group["message_ids"])
                current_group["ts"] = group["ts"]
            else:
                merged_groups.append(current_group)
                current_group = group.copy()
        merged_groups.append(current_group)
        return merged_groups

    def _is_consecutive_message(
        self, current_group: Dict[str, Any], next_msg: Dict[str, Any]
    ) -> bool:
        """Check if next message is consecutive (within 60 minutes) to current group."""
        try:
            start_ts = float(current_group.get("start_ts"))
            next_ts = float(next_msg.get("ts"))

            # Consider consecutive if within 1 hour (3600 seconds)
            time_diff = next_ts - start_ts
            return 0 < time_diff <= 3600
        except (ValueError, TypeError):
            return False

    def _filter_unessential_words(self, text: str) -> str:
        """Remove greeting words, unessential words, and emoji patterns from text."""
        if not text:
            return ""

        text = self._clean_slack_text(text)
        text = re.sub(r":[\w+-]+:", "", text)  # Remove emoji patterns
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r"[.!?]\s+", text)
        filtered = [
            self._filter_sentence(s)
            for s in sentences
            if s.strip() and self._filter_sentence(s)
        ]
        return ". ".join(filtered).strip()

    def _filter_sentence(self, sentence: str) -> str:
        """Filter a single sentence, removing greeting/unessential words."""
        sentence = sentence.strip()
        if not sentence:
            return ""

        sentence_lower = sentence.lower()
        greeting_words = [w for w in self._greeting_words if w in sentence_lower]
        unessential_words = [w for w in self._unessential_words if w in sentence_lower]

        if greeting_words or unessential_words:
            for word in greeting_words + unessential_words:
                sentence_lower = sentence_lower.replace(word, "")

        if len(sentence_lower.strip().split()) <= 3:
            return ""

        return sentence_lower.strip()

    def _convert_messages_to_documents(
        self, grouped_messages: List[Dict[str, Any]]
    ) -> List[Document]:
        """Convert grouped Slack messages to Document objects."""
        documents = []

        for group in tqdm(
            grouped_messages, desc="Converting grouped messages to documents"
        ):
            try:
                doc = self._group_to_document(group)
                if doc and doc.page_content:
                    documents.append(doc)
            except Exception as e:
                logger.warning(
                    f"Error converting group {group.get('id', 'unknown')}: {e}"
                )

        return documents

    def _group_to_document(self, group: Dict[str, Any]) -> Optional[Document]:
        """
        Convert a grouped message (from filter_and_group_messages) to a Document.
        """
        text = group.get("text", "").strip()
        if not text or len(text) < 10:  # Skip very short messages
            return None

        message_id = group.get("id", "")
        message_ids = group.get("message_ids", [message_id] if message_id else [])
        channel_id = group.get("channel_id", "")
        user_name = group.get("user_name", "")
        ts = group.get("ts")
        thread_ts = group.get("thread_ts")
        is_grouped = group.get("is_grouped", False)
        team_id = group.get("team_id", "")

        return Document(
            page_content=text,
            metadata={
                "doc_id": ts,
                "team_id": team_id,
                "type": "slack",
                "channel_id": channel_id,
                "user_name": user_name or "unknown",
                "timestamp": ts,
                "is_grouped": is_grouped,
                "thread_ts": thread_ts if thread_ts else "",
                "group_size": len(message_ids),
            },
        )

    def _clean_slack_text(self, text: str) -> str:
        """Clean Slack-specific formatting from text."""
        # Remove user mentions <@U123456>
        text = re.sub(r"<@[A-Z0-9]+>", "", text)

        # Convert channel mentions <#C123456|channel-name> to #channel-name
        text = re.sub(r"<#([A-Z0-9]+)\|([^>]+)>", r"#\2", text)

        # Convert URLs <https://example.com|link text> to link text
        text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2", text)
        text = re.sub(r"<([^>]+)>", r"\1", text)

        # Remove emoji codes :emoji_name:
        text = re.sub(r":[\w+-]+:", "", text)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
