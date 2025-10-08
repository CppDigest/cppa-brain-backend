"""
MailJsonProcessor: process mail-related JSON files for RAG.

Features:
- Read JSON files from data/source_data/new/<lang>/mail_json
- Extract fields: message_id_hash, content.thread, content.subject, content.parent,
  content.children, content.content, content.sender.address
- Create new JSON files with the extracted fields plus a summary of content.content
  using the pipeline's summarizer if available
- Save processed files under data/processed/mail_json/<lang>

Note:
- Hierarchical graph building has been moved to rag/HybridRetriever. This class
  is now responsible only for extracting and saving normalized mail records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from loguru import logger

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_config
from data_processor.summarize_processor import SummarizePocessor


class MailJsonProcessor:
    def __init__(self, language: str, summarizer: Optional[Any] = None):
        self.language = language or get_config("language.default_language", "en")
        self.logger = logger.bind(name="MailJsonProcessor")
        self.input_dir = Path(get_config("data.source_data.new_data_path", "data/source_data/new")) / self.language / "mail_json"
        self.output_dir = Path(get_config("data.processed_data.mail_json_path", "data/processed/mail_json")) / self.language
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Summarizer
        self.summarizer = summarizer if summarizer is not None else self._default_summarizer()
        # Cache processed message IDs to avoid duplicates
        self.processed_ids: Set[str] = self._load_processed_ids()

    def _default_summarizer(self):
        try:
            model_name = get_config("rag.hierarchical.summarization_model", "csebuetnlp/mT5_multilingual_XLSum")
            return SummarizePocessor(model_name=model_name)
        except Exception:
            return None

    def _summarize(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) < 300:
            return text
        if self.summarizer and hasattr(self.summarizer, "summarize_2_3_sentences"):
            try:
                return self.summarizer.summarize_2_3_sentences(text)
            except Exception:
                pass
        # naive fallback: first 2-3 sentences
        import re
        parts = re.split(r"(?<=[\.!?])\s+", text)
        summary = " ".join([p.strip() for p in parts if p.strip()][:3])
        return summary if summary else text[:200]

    def _load_processed_ids(self) -> Set[str]:
        """Load message_id_hash values from existing processed output files."""
        ids: Set[str] = set()
        try:
            patterns = [
                "*_processed.json",
                "*_processed_*.json",
                "*_processed_part*.json",
                "*_processed_part_*.json",
            ]
            files: List[Path] = []
            for pattern in patterns:
                files.extend(self.output_dir.glob(pattern))

            for pf in files:
                try:
                    data = json.loads(Path(pf).read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict):
                            mid = rec.get("message_id_hash")
                            if mid:
                                ids.add(mid)
        except Exception as exc:
            self.logger.warning(f"Failed to load processed ids: {exc}")
        self.logger.info(f"Loaded {len(ids)} processed message IDs from {self.output_dir}")
        return ids

    def _extract_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        content_obj = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
        sender_obj = content_obj.get("sender", {}) if isinstance(content_obj.get("sender"), dict) else item.get("sender", {}) or {}

        message_id_hash = (
            item.get("message_id_hash")
            or item.get("messageIdHash")
            or item.get("message-id-hash")
        )
        subject = content_obj.get("subject") or item.get("subject")
        thread = content_obj.get("thread") or item.get("thread")
        parent = content_obj.get("parent") or item.get("parent")
        children = content_obj.get("children") or item.get("children") or []
        body = content_obj.get("content") or item.get("content") or ""
        sender_address = (
            sender_obj.get("address")
            or sender_obj.get("email")
            or sender_obj.get("mail")
        )

        summary = self._summarize(body)

        return {
            "message_id_hash": message_id_hash,
            "content": {
                "thread": thread,
                "subject": subject,
                "parent": parent,
                "children": children,
                "content": body,
                "sender": {"address": sender_address},
            },
            "summary": summary,
        }

    def save_records(self, records: List[Dict[str, Any]], infile: Path, split_no: int) -> Path:
        """Save records to a JSON file."""
        
        outfile = self.output_dir / f"{infile.stem}_processed_part_{split_no:03d}.json"
        try:
            outfile.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            self.logger.info(f"Processed {infile} -> {outfile} ({len(records)} records)")
            return outfile
        except Exception as exc:
            self.logger.error(f"Failed to write {outfile}: {exc}")
            return None

    def get_split_no(self, infile_stem: str) -> int:
        """Get the split number from the infile stem."""
        files = self.output_dir.glob(f"{infile_stem}_processed_part_*.json")
        if files:
            no_list = [int(file.stem.split("_")[-1]) for file in files]
            return max(no_list)
        return 0
    

    def process_inputs(self) -> List[Path]:
        """Process all input mail JSON files and write processed outputs.

        Returns list of processed output file paths.
        """
        try:
            max_per_file = int(get_config("data.processed_data.max_count_per_file", 99))
        except Exception:
            max_per_file = 99

        if not self.input_dir.exists():
            self.logger.info(f"No mail_json directory found: {self.input_dir}")
            return []

        outputs: List[Path] = []
        input_files = sorted(self.input_dir.glob("*.json"))
        if not input_files:
            self.logger.info(f"No JSON files found in {self.input_dir}")
            return []

        for infile in input_files:
            try:
                data = json.loads(infile.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.warning(f"Failed to read {infile}: {exc}")
                continue

            records: List[Dict[str, Any]] = []
            split_no = self.get_split_no(infile.stem)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        mid = (item.get("message_id_hash") or item.get("messageIdHash") or item.get("message-id-hash"))
                        if mid and mid in self.processed_ids:
                            continue
                        if len(records) >= max_per_file:
                            split_no += 1
                            outfile = self.output_dir / f"{infile.stem}_processed_part_{split_no:03d}.json"
                            outputs.append(self.save_records(records, infile, split_no))
                            records = []
                        records.append(self._extract_record(item))
                        if mid:
                            self.processed_ids.add(mid)
            elif isinstance(data, dict):
                # assume it contains a list under a common key or a single message
                if "messages" in data and isinstance(data["messages"], list):
                    for item in data["messages"]:
                        if isinstance(item, dict):
                            mid = (item.get("message_id_hash") or item.get("messageIdHash") or item.get("message-id-hash"))
                            if mid and mid in self.processed_ids:
                                continue
                            if len(records) >= max_per_file:
                                split_no += 1
                                
                                outputs.append(self.save_records(records, infile, split_no))
                                records = []
                            records.append(self._extract_record(item))
                            if mid:
                                self.processed_ids.add(mid)
                else:
                    mid = (data.get("message_id_hash") or data.get("messageIdHash") or data.get("message-id-hash"))
                    if not (mid and mid in self.processed_ids):
                        records.append(self._extract_record(data))
                        if mid:
                            self.processed_ids.add(mid)
            else:
                self.logger.warning(f"Unsupported JSON structure in {infile}")
                continue

            # save the last batch
            if len(records) > 0:
                split_no += 1
                outputs.append(self.save_records(records, infile, split_no))

        return outputs

    def build_hierarchical_graph(self, processed_files: Optional[List[Path]] = None) -> Optional[Path]:
        """Deprecated: Building mail hierarchy moved to HybridRetriever.

        Use `rag.hybrid_retriever.HybridRetriever.build_mail_hierarchy_graph` instead.
        This method is kept for backward compatibility and now does nothing.
        """
        self.logger.warning(
            "MailJsonProcessor.build_hierarchical_graph is deprecated. "
            "Use HybridRetriever.build_mail_hierarchy_graph instead."
        )
        return None


