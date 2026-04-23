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
    
    # Expanded stop words — must aggressively filter generic English at scale
    STOP_WORDS = {
        # Articles, pronouns, determiners, prepositions
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "day", "get", "has",
        "him", "his", "how", "its", "may", "new", "now", "old", "see",
        "way", "who", "did", "let", "say", "she", "too", "use",
        "this", "that", "these", "those", "when", "where", "which", "what",
        "there", "their", "after", "before", "will", "have", "with", "from",
        "about", "would", "could", "should", "some", "other", "each", "every",
        "been", "being", "were", "does", "done", "doing", "then", "than",
        "into", "over", "under", "again", "further", "here", "just", "also",
        "very", "more", "most", "much", "many", "well", "still", "only",
        "even", "back", "down", "such", "like", "make", "made", "take",
        "took", "come", "came", "give", "gave", "know", "knew", "think",
        "thought", "look", "looked", "want", "wanted", "tell", "told",
        "work", "working", "worked", "call", "called", "need", "needed",
        "keep", "kept", "feel", "felt", "seem", "seemed", "help", "helped",
        # Generic verbs/actions that match everything
        "using", "used", "create", "created", "creating", "include", "includes",
        "including", "implement", "implementing", "ensure", "ensuring",
        "handle", "handling", "provide", "providing", "consider", "considering",
        "require", "requires", "required", "requirements", "following",
        "different", "specific", "based", "example", "approach", "process",
        "system", "systems", "project", "information", "important",
        "allows", "support", "address", "manage", "perform", "update",
        "updated", "change", "changed", "added", "adding", "start", "started",
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
            self.original_docs = {}

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Sharpened ontological concept extraction.
        Multi-tier strategy for discriminating topic identification:
        - Tier 1: Technical compound terms (hyphenated, camelCase, snake_case)
        - Tier 2: Proper nouns and acronyms (capitalized, ALL-CAPS)
        - Tier 3: High-frequency bigrams and trigrams (adjacent noun phrases)
        - Tier 4: Significant single terms (4+ chars, not stop words)
        """
        concepts = []
        
        # Tier 1: Technical compounds — most discriminating
        # Captures: Flask-Login, Flask-SQLAlchemy, account_lockout, camelCase
        compounds = re.findall(r'\b[A-Za-z][\w]*[-_][A-Za-z][\w]*(?:[-_][A-Za-z][\w]*)*\b', text)
        concepts.extend(c.lower() for c in compounds)
        
        # Tier 2: Proper nouns (3+ chars capitalized) and acronyms (2+ ALL-CAPS)
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        concepts.extend(w.lower() for w in proper_nouns if w.lower() not in self.STOP_WORDS)
        concepts.extend(w.lower() for w in acronyms if len(w) >= 2 and w.lower() not in self.STOP_WORDS)
        
        # Tier 3: High-frequency bigrams and trigrams from significant words
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in self.STOP_WORDS and len(w) >= 4]
        # Bigrams
        for i in range(len(filtered) - 1):
            concepts.append(f"{filtered[i]}_{filtered[i+1]}")
        # Trigrams — capture multi-word domain phrases like "user_authentication_flow"
        for i in range(len(filtered) - 2):
            concepts.append(f"{filtered[i]}_{filtered[i+1]}_{filtered[i+2]}")
        
        # Tier 4: Significant single terms (4+ chars, not stop)
        singles = [w for w in words if len(w) >= 4 and w not in self.STOP_WORDS]
        concepts.extend(singles)
        
        # Deduplicate and rank by frequency (most frequent = most characteristic)
        from collections import Counter
        freq = Counter(concepts)
        ranked = [term for term, _ in freq.most_common()]
        return ranked[:15]  # Top 15 for richer discriminating topology

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
        connections.append(f"[doc_{doc.id}]")
        
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
            
            if not hasattr(self, "original_docs"):
                self.original_docs = {}
            if uid not in self.original_docs:
                self.original_docs[uid] = {}
            self.original_docs[uid][f"doc_{doc.id}".lower()] = doc

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

        query_concepts = set(self._extract_concepts(query))
        # Bridge topological concepts to raw text matching spaces instead of underscores
        query_terms = {c.replace('_', ' ') for c in query_concepts}
        
        scored_nodes = []

        # Search through all clusters/nodes in the user's graph
        for cluster in self.graphs.get(uid, []):
            
            # Step 1: Extract any doc IDs referenced in this cluster's entire topology
            cluster_doc_keys = set()
            def find_docs(block):
                for n in block.get("nodes", []):
                    action_str = str(n.get("action", "")).lower()
                    if action_str.startswith("doc_"):
                        cluster_doc_keys.add(action_str[4:])
                    for conn in n.get("data_connections", []):
                        conn_lower = str(conn).lower()
                        if conn_lower.startswith("d_doc_"):
                            cluster_doc_keys.add(conn_lower[6:])
                for sub in block.get("sub_blocks", []):
                    find_docs(sub)
            
            find_docs(cluster)
            
            # Step 2: Extract all nodes and compute scores
            def score_nodes(block):
                for node in block.get("nodes", []):
                    action = str(node.get("action", ""))
                    connections_raw = node.get("data_connections", [])
                    connections = str(connections_raw).lower()
                    logic = action + " " + " ".join(connections_raw)
                    
                    logic_words = set(re.findall(r'\b\w+\b', logic.lower()))
                    action_words = set(re.findall(r'\b\w+\b', action.lower()))
                    
                    score = 0
                    for term in query_terms:
                        if term in logic_words:
                            score += 5
                        if term in action_words:
                            score += 2
                    
                    for concept in query_concepts:
                        if concept.lower() in connections:
                            score += 10
                            
                    if score > 0:
                        scored_nodes.append((score, node, cluster_doc_keys))
                        
                for sub in block.get("sub_blocks", []):
                    score_nodes(sub)

            score_nodes(cluster)

        # Sort by score desc and take top k
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_nodes[:k * 2]

        # Pre-compute inverse document frequency for query terms across user corpus
        user_docs_dict = self.original_docs.get(uid, {})
        num_docs = len(user_docs_dict)
        
        # Adaptive turn selection: fewer turns per doc when corpus is large
        # so total context stays within LLM's effective reasoning window
        if num_docs <= 10:
            turns_per_doc = 12   # Small corpus — keep broad context
        elif num_docs <= 30:
            turns_per_doc = 8    # Medium corpus — moderate extraction
        else:
            turns_per_doc = 6    # Large corpus — surgical extraction
        
        term_doc_count = {}  # How many docs contain each query term
        term_doc_freq = {}   # Term frequency per doc
        
        for doc_key, doc in user_docs_dict.items():
            doc_lower = doc.content.lower()
            term_doc_freq[doc_key] = {}
            for term in query_terms:
                count = doc_lower.count(term)
                if count > 0:
                    term_doc_count[term] = term_doc_count.get(term, 0) + 1
                    term_doc_freq[doc_key][term] = count
        
        import math
        
        # Calculate how specific/rare the query is globally
        query_max_idf = 1.0
        if query_terms:
            idf_vals = [math.log((num_docs + 1) / (term_doc_count.get(t, 1) + 1)) + 1 for t in query_terms]
            if idf_vals:
                query_max_idf = max(idf_vals)
                
        # Dynamic multiplier: factual queries (rare terms) isolate docs purely by TF-IDF (multiplier 3000+)
        # Reasoning queries (common terms) allow topological graphs to assist (multiplier < 200)
        dynamic_multiplier = 40 * (query_max_idf ** 3)
        
        doc_scores = {}
        for doc_key, doc in user_docs_dict.items():
            
            # Signal 0: Document-level TF-IDF score (BM25-like discriminator)
            tfidf_score = 0.0
            for term in query_terms:
                tf = term_doc_freq.get(doc_key, {}).get(term, 0)
                if tf > 0:
                    df = term_doc_count.get(term, 1)
                    idf = math.log((num_docs + 1) / (df + 1)) + 1
                    # Sublinear TF scaling, no length penalty for software docs
                    tfidf_score += (1 + math.log(1 + tf)) * idf
            
            # Phase 1: Turn-level splitting (preserves conversational boundaries)
            doc_turns = re.split(r'\n(?=\[?(?:Turn|[A-Z][a-z]+-\d+-\d{4} \| Turn) )', doc.content)
            turn_scores = []
            
            for turn_idx, turn in enumerate(doc_turns):
                turn_lower = turn.lower()
                turn_score = 0
                
                # Signal 1: Topological node intersection
                for score, node, _ in top_k:
                    action = str(node.get("action", "")).lower()
                    
                    if action and len(action) > 2 and action in turn_lower:
                        turn_score += score
                    
                    for conn in node.get("data_connections", []):
                        conn_str = str(conn).lower()
                        if conn_str.startswith("d_") or conn_str.startswith("f_") or conn_str.startswith("a_"):
                            conn_str = conn_str[2:]
                        if conn_str.startswith("doc_"):
                            continue
                            
                        if conn_str and len(conn_str) > 2 and conn_str in turn_lower:
                            turn_score += (score * 0.5)
                
                # Signal 2: Direct query term presence — critical discriminator
                # at scale where topology nodes match broadly across many docs
                query_hit_count = 0
                turn_tfidf_score = 0.0
                for t in query_terms:
                    if t in turn_lower:
                        query_hit_count += 1
                        df = term_doc_count.get(t, 1)
                        idf = math.log((num_docs + 1) / (df + 1)) + 1
                        
                        # Intra-document Inverse Term Frequency (ITF)
                        # Massively boosts turns containing terms that are rare inside this specific document
                        tf_global = term_doc_freq.get(doc_key, {}).get(t, 0)
                        intra_idf = 1000.0 / (1.0 + tf_global)
                        
                        turn_tfidf_score += (idf * intra_idf)
                        
                if query_hit_count >= 1:
                    turn_score += turn_tfidf_score * max(10, dynamic_multiplier)
                    
                turn_scores.append((turn_score, turn_idx, turn))
            
            # Phase 2: Select top turns + neighbors
            if turn_scores:
                scored_only = [(s, idx, t) for s, idx, t in turn_scores if s > 0]
                if not scored_only:
                    # Fallback: if no specific turn matched strongly but doc was 
                    # retrieved via TF-IDF, just take the first N turns.
                    scored_only = [(0.1, idx, t) for idx, t in enumerate(doc_turns[:turns_per_doc])]
                    if not scored_only:
                        continue
                    
                ranked = sorted(scored_only, key=lambda x: x[0], reverse=True)
                
                selected_indices = set()
                for _, idx, _ in ranked[:turns_per_doc]:
                    selected_indices.add(idx)
                    if idx > 0:
                        selected_indices.add(idx - 1)
                    if idx < len(doc_turns) - 1:
                        selected_indices.add(idx + 1)
                
                # Always anchor first turn
                selected_indices.add(0)
                
                # Phase 3: Sub-trim only very long turns at paragraph level
                selected_sorted = sorted(selected_indices)
                trimmed_turns = []
                for i in selected_sorted:
                    turn = doc_turns[i]
                    paragraphs = re.split(r'\n\n+', turn)
                    
                    if len(paragraphs) <= 6:
                        # Normal turns — keep as-is
                        trimmed_turns.append(turn)
                    else:
                        # Oversized turns — trim irrelevant filler paragraphs
                        kept = [paragraphs[0]]  # Always keep the turn header
                        for para in paragraphs[1:]:
                            para_lower = para.lower()
                            q_hits = sum(1 for t in query_terms if t in para_lower)
                            topo_hit = any(
                                str(node.get("action", "")).lower() in para_lower
                                for _, node, _ in top_k[:5]
                            )
                            if q_hits >= 1 or topo_hit or len(para) < 300:
                                kept.append(para)
                        trimmed_turns.append("\n\n".join(kept))
                
                # Hybrid scoring: topology turn-score + document-level TF-IDF
                topo_score = sum(s for s, _, _ in ranked[:turns_per_doc])
                
                if num_docs > 1:
                    # Multi-document splits (100k, 500k, 1M): Strict Lexical Supremacy
                    # Increase multiplier to 5000 to completely dominate polynomial topology
                    # score accumulation, ensuring precise factual matching dictates top-K docs.
                    doc_score = tfidf_score * 5000 + topo_score
                else:
                    # Mega-document splits (10M): Single 14.5MB connected string.
                    # Rely on the dynamic ITF-weighted turn selection and topological ties
                    doc_score = tfidf_score * dynamic_multiplier + topo_score
                
                synthesized_content = "\n\n...[...]\n\n".join(trimmed_turns)
                new_doc = Document(id=doc.id, content=synthesized_content, user_id=doc.user_id)
                doc_scores[doc_key] = (doc_score, new_doc)
                
        # Sort documents by their cumulative topological intersection score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        
        results = [doc for _, doc in sorted_docs[:k]]
        # Absolute fallback if query had zero intersection with any graph concepts
        if not results:
            for score, node, _ in top_k[:k]:
                results.append(Document(
                    id=node.get("id", "unknown"),
                    content=node.get("action", ""),
                    user_id=uid
                ))
        
        return results[:k], {"total_nodes_searched": sum(len(c.get("nodes", [])) for c in self.graphs.get(uid, []))}
