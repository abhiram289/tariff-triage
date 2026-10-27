import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.retrieval.indexer import CorpusIndexer


class RetrievalTools:
    def __init__(self, index_dir: str = settings.VECTOR_DB_PATH):
        self.indexer = CorpusIndexer(index_dir=index_dir)

    def search_corpus(
        self,
        query: str,
        top_k: int = 4,
        document_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tool 1: Semantic vector similarity search in the regulatory corpus.
        Returns top matching passages with clause names, document source, and page numbers.
        """
        try:
            query_embedding = self.indexer.embedder.encode([query], convert_to_numpy=True).tolist()
            where_clause = {"document_name": document_filter} if document_filter else None

            results = self.indexer.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            passages = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc_text, meta, dist in zip(docs, metas, dists):
                    passages.append({
                        "document": meta.get("document_name", "Unknown"),
                        "clause": meta.get("clause", "General"),
                        "page_number": meta.get("page_number", 1),
                        "similarity_score": round(1.0 - float(dist), 4),
                        "text": doc_text
                    })
            return passages
        except Exception as e:
            return [{"error": f"Corpus search failed: {str(e)}"}]

    def lookup_clause(self, clause_identifier: str) -> List[Dict[str, Any]]:
        """
        Tool 2: Keyword and clause identifier lookup in the indexed regulatory corpus.
        Searches metadata clause tags or exact phrases like 'LT-2(a)', 'mixed use', 'tailoring', 'Clause 4'.
        """
        try:
            # Query chunks where clause contains identifier, or text contains phrase
            results = self.indexer.collection.get(
                where={"clause": {"$contains": clause_identifier}},
                limit=5,
                include=["documents", "metadatas"]
            )

            # If metadata search found few, fallback to broad query
            if not results or not results.get("documents") or len(results["documents"]) == 0:
                return self.search_corpus(query=clause_identifier, top_k=3)

            passages = []
            for doc_text, meta in zip(results["documents"], results["metadatas"]):
                passages.append({
                    "document": meta.get("document_name", "Unknown"),
                    "clause": meta.get("clause", "General"),
                    "page_number": meta.get("page_number", 1),
                    "text": doc_text
                })
            return passages
        except Exception as e:
            return [{"error": f"Clause lookup failed: {str(e)}"}]

    @staticmethod
    def lookup_request_fields(dispute_data: Dict[str, Any], field_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Tool 3: Dynamic lookup of the consumer request's own fields.
        Allows the agent to inspect specific parameters on demand.
        """
        if not field_name or field_name == "all":
            return dispute_data
        return {field_name: dispute_data.get(field_name, "Field not found in request")}
