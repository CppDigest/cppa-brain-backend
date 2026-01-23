"""
Documentation preprocessing pipeline for LangChain RAG
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
import re
from tqdm import tqdm
from datetime import datetime

from config import DocuConfig


class DocuPreprocessor:
    """Process Boost library documentation for RAG"""

    def __init__(
        self,
    ):
        """
        Initialize Documentation preprocessor.

        Args:
            docu_config: DocuConfig instance (loads from env vars if not provided)
        """
        self.docu_config = DocuConfig()
        self.raw_data_dir = Path(self.docu_config.raw_data_dir)
        self.md_data_dir = Path(self.docu_config.md_data_dir)

    def load_documents(self) -> List[Document]:
        """Load and process all documents from data directory"""
        docs_path = self.md_data_dir
        if docs_path.exists():
            return self._get_documents_from_markdown_path(docs_path)
        return []

    def _get_documents_from_markdown_path(self, docs_path: Path) -> List[Document]:
        documents = []
        build_time = datetime.now().timestamp()

        for file_path in tqdm(docs_path.rglob("*.md"), desc="Processing documentation"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            url, content = self._extract_url_and_content(file_path, content)
            if len(content) < 50:
                continue
            lib_or_field = self._extract_library_from_path(file_path)
            doc = Document(
                page_content=content,
                metadata={
                    "doc_id": url,
                    "lang": "en",
                    "type": "documentation",
                    "library": lib_or_field,
                    "version": "",
                    "build_time": build_time,
                },
            )
            documents.append(doc)
        return documents

    def _extract_library_from_path(self, file_path: Path) -> str:
        """Extract library/source name from file path"""
        try:
            parts = file_path.parts
            for i, part in enumerate(parts):
                if "documentation" in part.lower() and i + 1 < len(parts):
                    return parts[i + 1]
            return parts[0] if parts else "unknown"
        except Exception:
            return "unknown"

    def _extract_url_and_content(
        self, file_path: Path, content: str
    ) -> tuple[str, str]:
        """Get URL from file path"""
        lines = content.split("\n")
        url = ""
        for line in lines[:3]:
            if "Source URL:" in line:
                url = line.split("Source URL:")[1].strip()
                break
            elif "**Source:**" in line:
                url = line.split("**Source:**")[1].strip()
                break

        if url == "":
            url = str(file_path).split("documentation")[-1].replace("\\", "/")
            url = url.replace(".html", "").replace("_", "/")
            if url.startswith("git_"):
                url = url.replace("git_", "https://github.com/")
            else:
                url = "https:/" + url
        else:
            content = "\n".join(lines[3:])
        content = re.sub(r"\n\s*\n+", "\n\n", content).strip()
        return url, content


def main():
    docu_preprocessor = DocuPreprocessor()
    documents = docu_preprocessor.load_documents()
    print(documents)


if __name__ == "__main__":
    main()
