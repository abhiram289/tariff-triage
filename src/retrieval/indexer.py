import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pypdf

from src.config import settings


class CorpusIndexer:
    def __init__(
        self,
        corpus_dir: str = settings.CORPUS_DIR,
        index_dir: str = settings.VECTOR_DB_PATH,
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME
    ):
        self.corpus_dir = Path(corpus_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model_name
        self.metadata_path = self.index_dir / "index_metadata.json"

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = None
        self._chroma_client = None
        self._collection = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(self.index_dir))
        return self._chroma_client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name="electricity_regulations",
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extracts text per page with clause and title awareness using pypdf."""
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for page_num, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({
                    "page_number": page_num + 1,
                    "document_name": pdf_path.name,
                    "text": text
                })
        return pages

    def chunk_page_text(
        self,
        page_dict: Dict[str, Any],
        chunk_size: int = 700,
        chunk_overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Splits text into chunks, preserving context and identifying section clauses.
        """
        text = page_dict["text"]
        doc_name = page_dict["document_name"]
        page_num = page_dict["page_number"]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []

        current_chunk = ""
        current_clause = "General"

        for p in paragraphs:
            # Check for clause / section patterns
            lines = p.splitlines()
            first_line = lines[0].strip() if lines else ""
            if any(term in first_line.lower() for term in ["clause", "section", "schedule", "tariff", "regulation", "lt-", "ht-"]):
                current_clause = first_line[:80]

            if len(current_chunk) + len(p) + 2 <= chunk_size:
                current_chunk += ("\n\n" + p if current_chunk else p)
            else:
                if current_chunk:
                    chunk_id = hashlib.md5(f"{doc_name}_{page_num}_{current_chunk[:50]}".encode("utf-8")).hexdigest()
                    chunks.append({
                        "id": chunk_id,
                        "document_name": doc_name,
                        "clause": current_clause,
                        "page_number": page_num,
                        "text": current_chunk
                    })
                # Overlap logic
                overlap = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = (overlap + "\n\n" + p).strip()

        if current_chunk:
            chunk_id = hashlib.md5(f"{doc_name}_{page_num}_{current_chunk[:50]}".encode("utf-8")).hexdigest()
            chunks.append({
                "id": chunk_id,
                "document_name": doc_name,
                "clause": current_clause,
                "page_number": page_num,
                "text": current_chunk
            })

        return chunks

    def index_corpus(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Idempotent indexing pipeline: only re-embeds new or changed chunks.
        Records data freshness and throughput metrics (D2).
        """
        start_time = time.time()
        pdf_files = list(self.corpus_dir.glob("*.pdf"))

        if not pdf_files:
            return {
                "status": "error",
                "message": f"No PDF files found in corpus directory: {self.corpus_dir}"
            }

        existing_count = self.collection.count()
        if force_reindex and existing_count > 0:
            self.chroma_client.delete_collection("electricity_regulations")
            self._collection = None

        all_chunks = []
        file_hashes = {}

        for pdf_path in pdf_files:
            with open(pdf_path, "rb") as f:
                f_hash = hashlib.sha256(f.read()).hexdigest()
            file_hashes[pdf_path.name] = {
                "sha256": f_hash,
                "size_bytes": pdf_path.stat().st_size,
                "modified_at": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
            }

            pages = self.extract_text_from_pdf(pdf_path)
            for page in pages:
                chunks = self.chunk_page_text(page)
                all_chunks.extend(chunks)

        total_chunks = len(all_chunks)
        if total_chunks == 0:
            return {"status": "warning", "message": "No chunks could be extracted from PDFs."}

        # Query existing IDs to maintain idempotency (D2: re-embedding without rebuilding whole index)
        existing_ids = set()
        try:
            get_res = self.collection.get(include=[])
            if get_res and "ids" in get_res:
                existing_ids = set(get_res["ids"])
        except Exception:
            pass

        new_chunks = [c for c in all_chunks if c["id"] not in existing_ids]

        if new_chunks:
            batch_size = 64
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i : i + batch_size]
                texts = [c["text"] for c in batch]
                embeddings = self.embedder.encode(texts, convert_to_numpy=True).tolist()
                ids = [c["id"] for c in batch]
                metadatas = [{
                    "document_name": c["document_name"],
                    "clause": c["clause"],
                    "page_number": c["page_number"]
                } for c in batch]

                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

        duration = max(time.time() - start_time, 0.001)
        throughput = round(len(new_chunks) / duration, 2)

        metadata = {
            "last_indexed_at": datetime.utcnow().isoformat(),
            "embedding_model": self.embedding_model_name,
            "source_files": file_hashes,
            "total_chunks_in_index": self.collection.count(),
            "new_chunks_added": len(new_chunks),
            "indexing_duration_sec": round(duration, 3),
            "chunks_per_second": throughput,
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def get_freshness_metrics(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            return {
                "is_indexed": False,
                "total_chunks": 0,
                "last_indexed_at": None,
                "throughput_chunks_per_sec": 0.0
            }
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "is_indexed": True,
            "total_chunks": meta.get("total_chunks_in_index", 0),
            "last_indexed_at": meta.get("last_indexed_at"),
            "throughput_chunks_per_sec": meta.get("chunks_per_second", 0.0)
        }


if __name__ == "__main__":
    force = "--force" in sys.argv
    print(f"Starting corpus indexing (force={force})...")
    indexer = CorpusIndexer()
    res = indexer.index_corpus(force_reindex=force)
    print(json.dumps(res, indent=2))
