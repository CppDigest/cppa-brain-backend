"""
WG21 paper preprocessing pipeline for LangChain RAG
"""

import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from tqdm import tqdm
from loguru import logger

from config import WG21Config
from preprocessor.utility import (
    get_timestamp_from_date,
    validate_content_length,
)


class WG21PaperPreprocessor:
    """Process WG21 C++ standard papers for RAG"""

    def __init__(self, wg21_config: Optional[WG21Config] = None):
        """
        Initialize WG21 paper preprocessor.

        Args:
            wg21_config: WG21Config instance (loads from env vars if not provided)
        """
        self.wg21_config = wg21_config or WG21Config()
        self.data_dir = Path(self.wg21_config.data_dir)
        self.metadata_file = self.data_dir / "wg21_papers_metadata_new.csv"
        self.logger = logger.bind(name="WG21PaperPreprocessor")

    def load_documents(self) -> List[Document]:
        """Load and process all WG21 papers from data directory"""
        if not self.data_dir.exists():
            self.logger.warning(f"WG21 data directory not found: {self.data_dir}")
            return []

        if not self.metadata_file.exists():
            self.logger.warning(f"Metadata file not found: {self.metadata_file}")
            return []

        # Load metadata from CSV
        metadata_dict = self._load_metadata()
        if not metadata_dict:
            self.logger.warning("No metadata loaded from CSV")
            return []
        documents = self._build_documents(metadata_dict)

        return documents

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from CSV file into a dictionary keyed by local_path or filename"""
        metadata_dict = {}

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Use local_path as primary key
                    local_path = row.get("local_path", "").strip()
                    filename = row.get("filename", "").strip()

                    # Store by both local_path and filename for flexible lookup
                    if local_path:
                        metadata_dict[local_path] = row
                    if filename and filename != local_path:
                        metadata_dict[filename] = row

        except Exception as e:
            self.logger.error(f"Error loading metadata CSV: {e}")

        return metadata_dict

    def _build_documents(
        self, metadata_dict: Dict[str, Dict[str, Any]]
    ) -> List[Document]:
        """Build documents from metadata dictionary"""
        # Process markdown files
        documents = []
        md_files = list(self.data_dir.rglob("*.md"))
        asc_files = list(self.data_dir.rglob("*.asc"))
        txt_files = list(self.data_dir.rglob("*.txt"))

        self.logger.info(f"Found {len(md_files)} markdown files")
        self.logger.info(f"Found {len(asc_files)} asc files")
        self.logger.info(f"Found {len(txt_files)} txt files")
        all_files = md_files + asc_files + txt_files

        for file_path in tqdm(all_files, desc="Processing WG21 papers"):
            try:
                # Get relative path from data_dir
                relative_path = file_path.relative_to(self.data_dir)
                relative_path_str = str(relative_path).replace("\\", "/")

                # Find matching metadata
                metadata = metadata_dict.get(relative_path_str) or metadata_dict.get(
                    str(relative_path)
                )

                # If no exact match, try matching by filename
                if not metadata:
                    filename = file_path.name
                    metadata = metadata_dict.get(filename)

                if not metadata:
                    self.logger.warning(f"No metadata found for {file_path}")
                    continue

                # Read markdown content
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not validate_content_length(content, min_length=50):
                    self.logger.warning(f"Skipping short file: {file_path}")
                    continue

                # Create document
                doc = self._create_document(content, metadata, file_path)
                if doc:
                    documents.append(doc)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        self.logger.info(f"Processed {len(documents)} WG21 paper documents")
        return documents

    def _create_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]],
        file_path: Path,
    ) -> Optional[Document]:
        """Create a Document from paper content and metadata"""

        # Extract document number from content if available
        doc_number = file_path.stem

        # Get metadata fields
        url = metadata.get("url", "") if metadata else ""
        author = metadata.get("author", "").strip() if metadata else ""
        authors = author.split(",") if author else ["unknown"]
        title = metadata.get("title", "").strip() if metadata else ""
        date_str = metadata.get("date", "") if metadata else ""

        # Parse date to timestamp
        timestamp = get_timestamp_from_date(date_str)
        if timestamp is None:
            self.logger.warning(f"Failed to parse date: {date_str}")
            return None

        # Build metadata
        doc_metadata = {
            "document_number": doc_number,
            "type": self.wg21_config.namespace,
            "title": title or file_path.stem,
            "author": authors,
            "timestamp": timestamp,
            "url": url,
        }

        return Document(page_content=content, metadata=doc_metadata)

    def _extract_document_number(self, content: str) -> Optional[str]:
        """Extract document number from paper content (e.g., P0843R10, N1234)"""
        patterns = [
            r"Document number[:\s]+([PN]\d+[A-Z]?\d*)",
            r"([PN]\d{4}[A-Z]?\d*)\s*[\.:]",
            r"paper\s+([PN]\d{4}[A-Z]?\d*)",
            r"proposal\s+([PN]\d{4}[A-Z]?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _generate_doc_id(
        self, url: str, file_path: Path, doc_number: Optional[str]
    ) -> str:
        """Generate document ID from URL, doc_number, or filename"""
        if doc_number:
            return f"wg21/{doc_number}"

        if url:
            match = re.search(r"/([PN]\d{4}[A-Z]?\d*)", url)
            if match:
                return f"wg21/{match.group(1)}"
            return url

        return f"wg21/{file_path.stem}"


def main():
    """Main function for testing"""
    preprocessor = WG21PaperPreprocessor()
    documents = preprocessor.load_documents()
    print(f"Loaded {len(documents)} documents")
    if documents:
        print("\nSample document metadata:")
        print(documents[0].metadata)


if __name__ == "__main__":
    main()
