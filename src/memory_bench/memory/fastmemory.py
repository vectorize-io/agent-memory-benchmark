from __future__ import annotations
import asyncio
import json
import logging
import re
import os
import sys
import fastmemory
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

from ..models import Document
from .base import MemoryProvider

logger = logging.getLogger(__name__)

class FastMemoryProvider(MemoryProvider):
    name = "fastmemory"
    description = "Topological Memory using NLTK concept extraction and Louvain graph clustering via a compiled Rust core."
    kind = "local"
    provider = "fastbuilder"
    link = "https://fastbuilder.ai"
    
    # Common words to ignore during concept extraction
    STOP_WORDS = {
        "this", "that", "these", "those", "when", "where", "which", "what", 
        "there", "their", "after", "before", "will", "have", "with", "from",
        "about", "would", "could", "should", "some", "other"
    }
    
    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.graphs: Dict[str, List[Dict[str, Any]]] = {}  # user_id -> compiled_graph
        self.concepts: Dict[str, Set[str]] = {}           # user_id -> global_concepts
        self.isolation_unit = "conversation"
        self.debug_mode = debug_mode or os.getenv("FM_DEBUG") == "1"
        self._engine_verified = False
        self._verify_engine_health()

    def _verify_engine_health(self):
        """Internal check to ensure the Rust engine and NLTK pipeline are working."""
        test_input = "The quick brown fox jumps over the lazy dog. Cats are independent animals."
        try:
            res = fastmemory.process_markdown(test_input)
            if res != "[]" and "block_type" in res:
                self._engine_verified = True
            else:
                self._print_engine_panic("Engine returned empty graph for test input.")
        except Exception as e:
            self._print_engine_panic(f"Engine crash: {str(e)}")

    def _print_engine_panic(self, detail: str):
        """Displays a diagnostic error for environment failures."""
        msg = f"""
################################################################################
#                                                                              #
#              FASTMEMORY ENGINE: INITIALIZATION FAILED                        #
#                                                                              #
################################################################################

DETAIL: {detail}

LIKELY CAUSES:
1. The embedded rust-louvain binary is not compatible with your platform.
   Run: file $(python3 -c "import fastmemory; print(fastmemory.__file__)")
2. NLTK data (punkt, averaged_perceptron_tagger_eng) is not installed.
   Run: python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

REMEDY:
- Upgrade: pip install --force-reinstall fastmemory>=0.4.3
- Verify: python3 scripts/verify_fastmemory.py

################################################################################
"""
        print(msg, file=sys.stderr)
        logger.critical(msg)

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        """Prepare local storage if needed. For now, we keep the graph in memory."""
        if reset:
            self.graphs = {}
            self.concepts = {}

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Lightweight entity/concept extraction.
        Identifies capitalized words and frequent nouns to build topological connections.
        """
        # Extract Capitalized Words (Proper Nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        
        # Extract potential concepts (words > 5 chars, not in stop words)
        words = re.findall(r'\b[a-z]{6,}\b', text.lower())
        concepts = [w for w in words if w not in self.STOP_WORDS]
        
        # Combine and unique
        all_concepts = list(set(proper_nouns + concepts))
        return list(all_concepts)[:5] # Limit to top 5 for dense connectivity

    def _sanitize_logic(self, content: str) -> str:
        """
        Sanitize content for Action-Topology Format (ATF).
        Escapes newlines and characters that confuse the Rust parser.
        """
        if not content:
            return ""
        # Escape newlines to prevent block termination
        content = content.replace("\r\n", " ").replace("\n", " ")
        # Escape quotes if necessary (ATF logic is space-delimited usually)
        content = content.replace('"', '\\"').strip()
        return content

    def _to_atf(self, doc: Document) -> str:
        """
        Convert a standard Document to Ontological ATF format.
        Builds 'Logic Rooms' based on extracted concepts.
        """
        sanitized_content = self._sanitize_logic(doc.content)
        concepts = self._extract_concepts(sanitized_content)
        
        # Build Data_Connections (Graph Edges)
        user_id = doc.user_id if doc.user_id else "default_user"
        connections = [f"[{user_id}]"]
        connections.extend([f"[{c}]" for c in concepts])
        
        # Dynamic Action name based on primary concept
        primary_concept = concepts[0] if concepts else "Standard"
        action_name = f"Process_{primary_concept}"
        
        # Action-Topology Format (ATF) wrapper
        return (
            f"## [ID: {doc.id}]\n"
            f"**Action:** {action_name}\n"
            f"**Input:** {{Data}}\n"
            f"**Logic:** {sanitized_content}\n"
            f"**Data_Connections:** {', '.join(connections)}\n"
            f"**Access:** Open\n"
            f"**Events:** Search\n\n"
        )

    def ingest(self, documents: List[Document]) -> None:
        """Ingest documents by compiling them into a topological logic graph."""
        # Group by user_id for isolation
        by_user: Dict[str, List[Document]] = {}
        for doc in documents:
            uid = doc.user_id if doc.user_id else "default_user"
            if uid not in by_user:
                by_user[uid] = []
            by_user[uid].append(doc)

        for uid, docs in by_user.items():
            atf_payload = "".join([self._to_atf(d) for d in docs])
            
            if self.debug_mode:
                print(f"\n--- [FM_DEBUG] ATF Payload for {uid} ---")
                print(atf_payload)
                print("--- [FM_DEBUG] END Payload ---\n")

            try:
                logger.info(f"Compiling FastMemory graph for user: {uid} ({len(docs)} docs)")
                json_graph_str = fastmemory.process_markdown(atf_payload)
                
                if os.environ.get("FM_DEBUG") == "1":
                    print(f"\n--- [FM_DEBUG] ATF Payload for {uid} ---\n{atf_payload}\n--- [FM_DEBUG] END Payload ---")
                    print(f"\n--- [FM_DEBUG] Raw Engine Return (len: {len(json_graph_str)}) ---\n{json_graph_str}\n--- [FM_DEBUG] END Engine ---")
                    if "Louvain" in json_graph_str:
                        print("--- [FM_DEBUG] Louvain clustering detected in engine output ---")

                if json_graph_str == "[]":
                    logger.error(f"FastMemory engine returned an empty graph for user {uid}.")
                    logger.error("DIAGNOSTIC: If you do not see '[Louvain]' logs above, the Rust engine failed to initialize.")
                    logger.error("Possible causes: (1) Python 3.9/3.10 binary mismatch (2) Missing macOS system libraries (3) Malformed ATF structure.")
                    continue
                
                graph_data = json.loads(json_graph_str)
                
                if uid not in self.graphs:
                    self.graphs[uid] = []
                
                # FastMemory returns a list of clusters (blocks)
                self.graphs[uid].extend(graph_data)
            except Exception as e:
                logger.error(f"FastMemory Ingestion Error for {uid}: {e}")
                if self.debug_mode:
                    print(f"!!! [FM_DEBUG] INGESTION ERROR: {e}")

    def retrieve(self, query: str, k: int = 10, user_id: str | None = None, query_timestamp: str | None = None) -> Tuple[List[Document], Dict | None]:
        """Retrieve top-k relevant documents using topological search."""
        uid = user_id if user_id else "default_user"
        if uid not in self.graphs or not self.graphs[uid]:
            if self.debug_mode:
                print(f"--- [FM_DEBUG] Search failed: Graph for user {uid} is empty. ---")
            return [], None

        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_terms = {w for w in query_words if w not in self.STOP_WORDS}
        query_concepts = set(self._extract_concepts(query))
        
        scored_nodes = []

        # Search through all clusters/nodes in the user's graph
        for cluster in self.graphs[uid]:
            for node in cluster.get("nodes", []):
                # Extract logic and metadata
                logic = node.get("logic", "").lower()
                node_id = node.get("id", "").lower()
                action = node.get("action", "").lower()
                
                # Data Connections (Topological Edges)
                # We prioritize nodes that share 'Concepts' with the query
                connections = str(node.get("data_connections", "")).lower()
                
                logic_words = set(re.findall(r'\b\w+\b', logic))
                action_words = set(re.findall(r'\b\w+\b', action))
                
                score = 0
                for term in query_terms:
                    if term in logic_words:
                        score += 1
                    if term in node_id:
                        score += 5  # High weight for ID matches (NIAH success)
                    if term in action_words:
                        score += 2
                
                # Topological Boost: If the query and node share a concept link
                for concept in query_concepts:
                    if concept.lower() in connections:
                        score += 10 # Massive boost for conceptual alignment
                
                if score > 0:
                    scored_nodes.append((score, node))

        # Sort by score desc and take top k
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_nodes[:k]

        results = []
        for score, node in top_k:
            results.append(Document(
                id=node.get("id", "unknown"),
                content=node.get("logic", ""),
                user_id=uid
            ))

        return results, {"total_nodes_searched": sum(len(c.get("nodes", [])) for c in self.graphs[uid])}
