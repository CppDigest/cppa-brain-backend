"""
Mail preprocessing pipeline for LangChain RAG
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from tqdm import tqdm
from loguru import logger
from datetime import datetime

from config import MailConfig
from preprocessor.utility import (
    get_timestamp_from_date,
    clean_text,
    validate_content_length,
)


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

    def convert_all_to_markdown(self) -> List[str]:
        """
        Convert all email threads to markdown files
        Returns:
            List of paths to saved markdown files
        """
        emails_path = self.mail_data_dir
        if not emails_path.exists():
            self.logger.warning(f"Mail data directory not found: {emails_path}")
            return []

        saved_files = []

        for json_file in tqdm(
            emails_path.rglob("*.json"), desc="Converting threads to markdown"
        ):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    mail_data = json.load(f)

                grouped_mail_data = self.group_by_thread(mail_data)

                for mails in grouped_mail_data.values():
                    thread_info = self.get_thread_info_from_content(mails)
                    if not thread_info:
                        self.logger.warning("Could not extract thread information")
                        continue
                    file_path = self._save_markdown_file(thread_info, mails)
                    if file_path:
                        saved_files.append(file_path)
            except Exception as e:
                self.logger.error(f"Error processing {json_file}: {e}")

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

            for message in mail_list:
                doc = self.build_document_for_pinecone(message)
                if doc:
                    documents.append(doc)

        except Exception as e:
            self.logger.error(f"Error processing mail thread: {e}")
        return documents

    def build_document_for_pinecone(self, message: Dict[str, Any]) -> Document:
        """Build document for Pinecone"""
        content = self._extract_message_content(message)
        if not content:
            return None

        thread_url = message.get("thread_url", message.get("thread_id", ""))
        thread_id = self._extract_id_from_url(thread_url)

        subject = message.get("subject", "No Subject")

        message_url = message.get("message_url", message.get("url", ""))
        if message_url == "":
            message_url = (
                f"{message.get("list_name", "")}/message/{message.get("msg_id", "")}"
            )

        msg_id = self._extract_id_from_url(message_url)

        parent_id = self._extract_id_from_url(
            message.get("parent", message.get("parent_id", ""))
        )
        timestamp = get_timestamp_from_date(
            message.get("date", message.get("sent_at", datetime.now().isoformat()))
        )

        return Document(
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

    def _extract_id_from_url(self, url: str) -> str:
        """Extract ID from URL"""
        if not url:
            return ""
        doc_id = url.split("list/")[-1].rstrip("/")
        doc_id = doc_id.replace("email", "message")
        return doc_id

    def _extract_message_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract and clean message content"""
        content = message.get("content", message.get("body", ""))
        if not content:
            return None

        # Clean up whitespace and quotes
        content = clean_text(content, remove_extra_spaces=True)

        return content if validate_content_length(content, min_length=20) else None

    def group_by_thread(self, mail_data: Any) -> Optional[str]:
        """Group email thread by thread_id"""
        try:
            mail_list, thread_info = self._extract_mail_data(mail_data)
            thread_info = {}
            for mail in mail_list:
                thread_id = mail.get("thread_id", "")
                if thread_id not in thread_info:
                    thread_info[thread_id] = []
                thread_info[thread_id].append(mail)

            for thread_id, mails in thread_info.items():
                mails = mails.sort(key=lambda x: x.get("sent_at", ""))

            return thread_info
        except Exception as e:
            self.logger.error(f"Error converting mail thread to markdown: {e}")
            return None

    def _extract_mail_data(self, mail_data: Any) -> tuple[List[Dict], Optional[Dict]]:
        """Extract mail list and thread info from mail data"""
        if isinstance(mail_data, list):
            return mail_data, None
        return mail_data.get("messages", []), mail_data.get("thread_info")

    def _build_markdown_header(self, thread_info: Dict[str, Any]) -> List[str]:
        """Build markdown header section"""
        list_name = thread_info.get("list_name", None)
        if not list_name:
            list_name = self.get_list_name_from_url(thread_info.get("url", ""))
        lines = [
            f"# LIST_NAME: {list_name}\n",
            f"# SUBJECT: {thread_info.get('subject', 'No Subject')}\n",
            f"**TYPE_ID:** thread/{thread_info.get('thread_id', '')}\n",
        ]
        if thread_info.get("sent_at"):
            lines.append(f"**DATE:** {thread_info.get('sent_at')}\n")
        lines.append("\n---\n\n")
        return "  \n".join(lines)

    def _build_message_section(
        self, idx: int, message: Dict[str, Any], content: str
    ) -> List[str]:
        """Build markdown section for a single message"""
        lines = [
            f"## Message {idx}\n\n",
            f"**From:** {message.get('sender_address', 'Unknown')}\n",
        ]
        if date := message.get("date", message.get("sent_at")):
            lines.append(f"**Date:** {date}\n")

        if message_id := message.get("msg_id"):
            lines.append(f"**TYPE_ID:** message/{message_id}\n")
        elif message_url := message.get("message_url", message.get("url")):
            if message_id := self.get_id_from_url(message_url):
                lines.append(f"**TYPE_ID:** {message_id}\n")

        if parent_id := message.get("parent_id"):
            lines.append(f"**In Reply To:** message/{parent_id}\n")
        elif parent := message.get("parent"):
            if parent_id := self.get_id_from_url(parent):
                lines.append(f"**In Reply To:** {parent_id}\n")

        lines.extend(["\n", f"{content}\n\n", "---\n\n"])

        return "  \n".join(lines)

    def _save_markdown_file(
        self, thread_info: Dict[str, Any], mails: str
    ) -> Optional[Path]:
        """Save markdown content to file"""
        output_dir = Path(self.mail_config.markdown_data_dir)
        output_path = (
            output_dir
            / f"emails_{thread_info.get("list_name", "").replace(
            "@", "_"
        ).replace(".", "_").replace("-", "_")}"
        )
        output_path.mkdir(parents=True, exist_ok=True)
        thread_id = thread_info.get("thread_id", "")
        file_path = output_path / f"thread_{thread_id}.md"
        content = self._build_markdown_header(thread_info)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")

        content = self._add_new_messages_to_content(content, mails)
        file_path.write_text(content, encoding="utf-8")
        self.logger.info(f"Saved markdown to {file_path}")
        return file_path

    def _add_new_messages_to_content(
        self, content: str, mails: List[Dict[str, Any]]
    ) -> str:
        """Add new messages to content
        ## Message 1

        **From:** jmihalicza@gmail.com
        **Date:** Sun, 09 Oct 2011 20:30:18 +0200
        **TYPE_ID:** message/ZZYO4DK7LLPW7QFXGV4R5HQRTZE7EKCL

        Hello, From: Domagoj Saric: ...you might want to try the Templight route. A couple of years back the Hungarian team behind it tried to do the same thing as yo
        """
        last_idx = len(content.split("## Message "))
        for mail in mails:
            msg_id = mail.get("msg_id", "")
            if f"message/{msg_id}" in content:
                continue

            content = content + self._build_message_section(
                last_idx, mail, mail.get("content", "")
            )
            last_idx += 1
        return content

    def get_list_name_from_url(self, url: str) -> str:
        """Get list name from URL"""
        list_name = ""
        try:
            list_name = url.split("/list/")[-1]
            list_name = list_name.split("/")[0]

        except Exception as e:
            self.logger.error(f"Error getting list name from URL: {e}")

        return list_name

    def get_id_from_url(self, url: str) -> Optional[str]:
        """Extract ID from URL with type prefix (thread/ or message/)"""
        if "/thread/" in url:
            return "thread/" + url.split("/thread/")[-1].rstrip("/")
        if "/email/" in url:
            return "message/" + url.split("/email/")[-1].rstrip("/")
        return None

    def get_thread_info_from_content(
        self, mails: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get thread information from content

        Args:
            thread_info: Existing thread info dict (optional)
            message: First message in thread (optional)

        Returns:
            Dictionary with thread information or None if insufficient data
        """
        thread_info = {}
        thread_info["thread_id"] = mails[0].get("thread_id", "")
        thread_info["list_name"] = mails[0].get("list_name", "")
        thread_info["subject"] = mails[0].get("subject", "")
        thread_info["sent_at"] = mails[0].get("sent_at", "")
        for mail in mails:
            thread_id = mail.get("thread_id", "")
            msg_id = mail.get("msg_id", "")
            if msg_id == thread_id:
                thread_info["subject"] = mail.get("subject", "")
                thread_info["sent_at"] = mail.get("sent_at", "")

        return thread_info


def main():
    mail_preprocessor = MailPreprocessor()
    mail_preprocessor.convert_all_to_markdown()


if __name__ == "__main__":
    main()
