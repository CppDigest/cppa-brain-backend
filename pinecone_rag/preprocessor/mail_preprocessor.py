"""
Mail preprocessing pipeline for LangChain RAG
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import re
from tqdm import tqdm
from loguru import logger
from datetime import datetime

from config import MailConfig


class MailPreprocessor:
    """Process Boost mailing list data for RAG"""

    def __init__(self, mail_config: Optional[MailConfig] = None):
        """
        Initialize Mail preprocessor.

        Args:
            mail_config: MailConfig instance (loads from env vars if not provided)
        """
        self.mail_config = mail_config or MailConfig()
        self.mail_data_dir = Path(self.mail_config.mail_data_dir)
        self.logger = logger.bind(name="MailPreprocessor")

    def load_emails_from_dir(self) -> List[Document]:
        """Load and process all emails from data directory"""
        emails_path = self.mail_data_dir
        if emails_path.exists():
            return self._process_mail_data(emails_path)
        return []

    def load_emails_from_mail_list(
        self, mail_list: List[Dict[str, Any]]
    ) -> List[Document]:
        """Process all emails from mail list"""
        return self.process_mail_list(mail_list)

    def convert_all_to_markdown(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Convert all email threads to markdown files

        Args:
            output_dir: Directory to save markdown files (default: mail_data_dir/markdown)

        Returns:
            List of paths to saved markdown files
        """
        emails_path = self.mail_data_dir
        if not emails_path.exists():
            self.logger.warning(f"Mail data directory not found: {emails_path}")
            return []

        saved_files = []

        for thread_file in tqdm(
            emails_path.rglob("*.json"), desc="Converting threads to markdown"
        ):
            try:
                with open(thread_file, "r", encoding="utf-8") as f:
                    mail_data = json.load(f)

                directory_name = thread_file.parent.name[0:-3].replace("-", "_")
                file_path = self.convert_to_markdown(
                    mail_data, output_dir + "/" + directory_name
                )
                if file_path:
                    saved_files.append(file_path)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON in {thread_file}: {e}")
            except Exception as e:
                self.logger.error(f"Error processing {thread_file}: {e}")

        self.logger.info(f"Converted {len(saved_files)} threads to markdown")
        return saved_files

    def _process_mail_data(self, mail_path: Path) -> List[Document]:
        """Process Boost mailing list data"""
        documents = []
        for thread_file in tqdm(
            mail_path.rglob("*.json"), desc="Processing mail threads"
        ):
            try:
                with open(thread_file, "r", encoding="utf-8") as f:
                    mail_data = json.load(f)
                documents.extend(self.process_mail_list(mail_data))
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON in {thread_file}: {e}")
            except Exception as e:
                self.logger.error(f"Error processing {thread_file}: {e}")
        self.logger.info(f"Processed {len(documents)} documents")
        return documents

    def process_mail_list(self, mail_data: Any) -> List[Document]:
        """Process individual mail thread"""
        documents = []
        try:
            mail_list = (
                mail_data
                if isinstance(mail_data, list)
                else mail_data.get("messages", [])
            )
            if not mail_list:
                return documents

            thread_url = mail_list[0].get("thread_url", "unknown")
            thread_id = self._extract_id_from_url(thread_url)
            subject = mail_list[0].get("subject", "No Subject")

            for message in mail_list:
                content = self._extract_message_content(message)
                if not content:
                    continue

                message_url = message.get("message_url", message.get("url", ""))
                msg_id = self._extract_id_from_url(message_url)
                parent_id = self._extract_id_from_url(message.get("parent", ""))
                timestamp = self.get_timestamp_from_date(
                    message.get("date", datetime.now().isoformat())
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "doc_id": msg_id,
                            "type": "mailing",
                            "thread_id": thread_id,
                            "subject": subject or "",
                            "author": message.get("sender_address", "") or "",
                            "timestamp": timestamp,
                            "parent_id": parent_id,
                        },
                    )
                )
        except Exception as e:
            self.logger.error(f"Error processing mail thread: {e}")
        return documents

    def _extract_id_from_url(self, url: str) -> str:
        """Extract ID from URL"""
        if not url:
            return ""
        doc_id = url.split("list/")[-1].rstrip("/")
        doc_id = doc_id.replace("email", "message")
        return doc_id

    def get_timestamp_from_date(self, date: str) -> float:
        """Get timestamp from date string"""
        if not date:
            return datetime.now().timestamp()

        formats = ["%Y-%m-%dT%H:%M:%SZ", "%a, %d %b %Y %H:%M:%S %z"]
        for fmt in formats:
            try:
                return datetime.strptime(date, fmt).timestamp()
            except ValueError:
                continue
        return datetime.now().timestamp()

    def _extract_message_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract and clean message content"""
        content = message.get("content", message.get("body", ""))
        if not content:
            return None

        # Clean up whitespace and quotes
        content = re.sub(r"\s+", " ", content).strip()
        content = content.replace('"', '"').replace('"', '"')

        return content if len(content) > 20 else None

    def convert_to_markdown(
        self, mail_data: Any, output_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert email thread to markdown and save as {thread_id}.md

        Args:
            mail_data: Email thread data (list of messages or dict with messages)
            output_dir: Directory to save markdown files

        Returns:
            Path to saved markdown file, or None if conversion failed
        """
        try:
            mail_list, thread_info = self._extract_mail_data(mail_data)
            if not mail_list:
                self.logger.warning("No messages found in mail data")
                return None

            thread_info = self.get_thread_info_from_content(thread_info, mail_list[0])
            if not thread_info:
                self.logger.warning("Could not extract thread information")
                return None

            markdown_content = self._build_markdown_content(thread_info, mail_list)
            file_path = self._save_markdown_file(
                thread_info.get("thread_id", "unknown"), markdown_content, output_dir
            )
            return str(file_path) if file_path else None
        except Exception as e:
            self.logger.error(f"Error converting mail thread to markdown: {e}")
            return None

    def _extract_mail_data(self, mail_data: Any) -> tuple[List[Dict], Optional[Dict]]:
        """Extract mail list and thread info from mail data"""
        if isinstance(mail_data, list):
            return mail_data, None
        return mail_data.get("messages", []), mail_data.get("thread_info")

    def _build_markdown_content(
        self, thread_info: Dict[str, Any], mail_list: List[Dict]
    ) -> str:
        """Build markdown content from thread info and messages"""
        lines = self._build_markdown_header(thread_info)
        for idx, message in enumerate(mail_list, 1):
            content = self._extract_message_content(message)
            if content:
                lines.extend(self._build_message_section(idx, message, content))
        return "".join(lines)

    def _build_markdown_header(self, thread_info: Dict[str, Any]) -> List[str]:
        """Build markdown header section"""
        list_name = self.get_list_name_from_url(thread_info.get("url", ""))
        lines = [
            f"# LIST_NAME: {list_name}\n",
            f"# SUBJECT: {thread_info.get('subject', 'No Subject')}\n",
            f"**TYPE_ID:** {thread_info.get('thread_id', 'unknown')}\n",
        ]
        if thread_info.get("date_active"):
            lines.append(f"**DATE:** {thread_info.get('date_active')}\n")
        lines.append("\n---\n\n")
        return lines

    def _build_message_section(
        self, idx: int, message: Dict[str, Any], content: str
    ) -> List[str]:
        """Build markdown section for a single message"""
        lines = [
            f"## Message {idx}\n\n",
            f"**From:** {message.get('sender_address', 'Unknown')}\n",
        ]
        if date := message.get("date"):
            lines.append(f"**Date:** {date}\n")
        if message_url := message.get("message_url", message.get("url")):
            if message_id := self.get_id_from_url(message_url):
                lines.append(f"**TYPE_ID:** {message_id}\n")
        if parent := message.get("parent"):
            if parent_id := self.get_id_from_url(parent):
                lines.append(f"**In Reply To:** {parent_id}\n")
        lines.extend(["\n", f"{content}\n\n", "---\n\n"])
        return lines

    def _save_markdown_file(
        self, thread_id: str, content: str, output_dir: Optional[str]
    ) -> Optional[Path]:
        """Save markdown content to file"""
        output_path = (
            Path(output_dir) if output_dir else self.mail_data_dir / "markdown"
        )
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{thread_id.replace('/', '_')}.md"
        file_path.write_text(content, encoding="utf-8")
        self.logger.info(f"Saved markdown to {file_path}")
        return file_path

    def get_list_name_from_url(self, url: str) -> str:
        """Get list name from URL"""
        list_name = url.split("/list/")[1]
        list_name = list_name.split("/")[0]
        return list_name

    def get_id_from_url(self, url: str) -> Optional[str]:
        """Extract ID from URL with type prefix (thread/ or message/)"""
        if "/thread/" in url:
            return "thread/" + url.split("/thread/")[-1].rstrip("/")
        if "/email/" in url:
            return "message/" + url.split("/email/")[-1].rstrip("/")
        return None

    def get_thread_info_from_content(
        self, thread_info: Any, message: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get thread information from content

        Args:
            thread_info: Existing thread info dict (optional)
            message: First message in thread (optional)

        Returns:
            Dictionary with thread information or None if insufficient data
        """
        if not thread_info and not message:
            return None

        result = thread_info.copy() if isinstance(thread_info, dict) else {}
        if not message:
            return (
                result
                if result.get("thread_id") and result.get("thread_id") != "unknown"
                else None
            )

        thread_url = message.get("thread_url", "")
        thread_id = self.get_id_from_url(thread_url) or "unknown"
        date_active = self._parse_date(message.get("date", ""))

        result.update(
            {
                "url": thread_url,
                "thread_id": thread_id,
                "subject": message.get("subject", result.get("subject", "No Subject")),
                "date_active": date_active or result.get("date_active", ""),
                "starting_email": message.get("url", message.get("message_url", "")),
            }
        )

        return (
            result
            if result.get("thread_id") and result.get("thread_id") != "unknown"
            else None
        )

    def _parse_date(self, date_str: str) -> str:
        """Parse date string to ISO format"""
        if not date_str:
            return ""
        formats = ["%Y-%m-%dT%H:%M:%SZ", "%a, %d %b %Y %H:%M:%S %z"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).isoformat()
            except (ValueError, AttributeError):
                continue
        return date_str


def main():
    mail_preprocessor = MailPreprocessor()
    mail_preprocessor.convert_all_to_markdown(
        output_dir="data/message_by_thread/markdown"
    )


if __name__ == "__main__":
    main()
