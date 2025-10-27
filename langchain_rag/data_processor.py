"""
Data processing pipeline for LangChain RAG
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import markdown
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import hashlib


class BoostDataProcessor:
    """Process Boost library documentation and mail data for RAG"""

    def __init__(
        self, 
        mail_data_dir: str = "data/processed/message_by_thread", 
        doc_data_dir: str = "data/source_data/processed/en", 
        chunk_size: int = 512, 
        chunk_overlap: int = 50
    ):
        self.mail_data_dir = Path(mail_data_dir)
        self.doc_data_dir = Path(doc_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_documents(self) -> List[Document]:
        """Load and process all documents from data directory"""
        documents = []

        # Process documentation files
        docs_path = self.doc_data_dir
        if docs_path.exists():
            docs = self._process_documentation(docs_path)
            documents.extend(docs)

        # Process mail data
        mail_path = self.mail_data_dir
        if mail_path.exists():
            mails = self._process_mail_data(mail_path)
            documents.extend(mails)

        return documents

    def _process_documentation(self, docs_path: Path) -> List[Document]:
        """Process Boost documentation files"""
        documents = []

        for file_path in tqdm(docs_path.rglob("*"), desc="Processing documentation"):
            if file_path.is_file() and file_path.suffix in [".txt", ".md", ".html"]:
                try:
                    content = self._read_file(file_path)
                    if content:
                        first_line = content.split("\n")[0]
                        if "Source URL:" in first_line:
                            url = first_line.split("Source URL:")[1].strip()
                            source = "Boost Documentation"
                        else:
                            url = str(file_path.relative_to(docs_path))
                            source = "github.com/boostorg"
                        doc_id = hashlib.md5(url.encode()).hexdigest()
                        doc = Document(
                            page_content=content.replace(first_line, ""),
                            id=doc_id,
                            metadata={
                                "source": source,
                                "type": "documentation",
                                "library": self._extract_library_name(file_path),
                                "file_type": file_path.suffix,
                                "url": url,
                                "version": "1.89.0"
                            },
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return documents

    def _process_mail_data(self, mail_path: Path) -> List[Document]:
        """Process Boost mailing list data"""
        documents = []

        for thread_file in tqdm(mail_path.rglob("*.json"), desc="Processing mail threads"):
            try:
                with open(thread_file, "r", encoding="utf-8") as f:
                    mail_data = json.load(f)
                thread_docs = self.process_mail_list(mail_data)
                documents.extend(thread_docs)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {thread_file}: {e}")
            except Exception as e:
                print(f"Error processing {thread_file}: {e}")

        return documents

    def process_mail_list(self, mail_data: Dict[str, Any]) -> List[Document]:
        """Process individual mail thread"""
        documents = []

        try:
            if isinstance(mail_data, list):
                mail_list = mail_data
            else:
                mail_list = mail_data.get("messages", [])

            # Extract thread information
            thread_url = mail_list[0].get("thread_url", "unknown") if mail_list else "unknown"
            subject = mail_list[0].get("subject", "No Subject") if mail_list else "No Subject"

            # Process each message in the thread
            for message in mail_list:
                content = self._extract_message_content(message)
                msg_id = message.get("message_id", "Unknown")
                if "@@" not in msg_id:
                    msg_id = f"@@MailingList@@{msg_id}"
                if content:
                    message_url = message.get("message_url", message.get("url", "Unknown"))
                    doc_id = hashlib.md5(message_url.encode()).hexdigest()
                    doc = Document(
                        page_content=content,
                        id=doc_id,
                        metadata={
                            "source": msg_id,
                            "type": "mail",
                            "thread_id": thread_url,
                            "subject": subject,
                            "author": message.get("sender_address", "Unknown"),
                            "date": message.get("date", "Unknown"),
                            "url": message_url,
                        },
                    )
                    documents.append(doc)

        except Exception as e:
            print(f"Error processing mail thread: {e}")

        return documents

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read and clean file content"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Clean content based on file type
            if file_path.suffix == ".html":
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text()
            elif file_path.suffix == ".md":
                # Convert markdown to plain text
                html_content = markdown.markdown(content)
                soup = BeautifulSoup(html_content, "html.parser")
                content = soup.get_text()

            # Clean up whitespace
            # content = re.sub(r"\s+", " ", content).strip()
            content = re.sub(r"\n\s*\n+", "\n\n", content).strip()
            return content if len(content) > 50 else None

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def _extract_library_name(self, file_path: Path) -> str:
        """Extract Boost library name from file path"""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part == "en" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"

    def _extract_message_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract and clean message content"""
        content = message.get("content", message.get("body", ""))
        if not content:
            return None

        # Clean up whitespace and quotes
        content = re.sub(r"\s+", " ", content).strip()
        content = content.replace('"', '"').replace('"', '"')

        return content if len(content) > 20 else None

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunked_documents = []
        for doc in documents:
            doc.id = doc.id if doc.id else hashlib.md5(doc.metadata["url"].encode()).hexdigest()
            chunks = self.text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                if len(chunk.page_content) < 50:
                    continue
                chunk.id = f"{doc.id}-{i:03d}"
                chunked_documents.append(chunk)
            
        return chunked_documents

