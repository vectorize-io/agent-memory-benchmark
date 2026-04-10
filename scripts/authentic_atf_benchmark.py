import os
import time
import json
import re
import string
import pandas as pd
from datasets import load_dataset
import fastmemory
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from huggingface_hub import hf_hub_download

# Ensure required NLTK packages are available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download issues: {e}")

STOP_WORDS = {"this", "that", "these", "those", "when", "where", "which", "what", "there", "their", "after", "before", "will", "have", "with", "from", "assistant", "user"}

def extract_concepts(text):
    """Entity/Concept extraction for topological linking."""
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        nouns = [word.lower() for (word, pos) in tagged if pos.startswith('NN') and word.lower() not in STOP_WORDS]
        proper_nouns = [word for (word, pos) in tagged if pos == 'NNP']
        return list(set(nouns[:3] + proper_nouns[:2]))
    except:
        words = text.translate(str.maketrans('', '', string.punctuation)).split()
        return [w.lower() for w in words if len(w) > 5 and w.lower() not in STOP_WORDS][:5]

def generate_atfs(segments, conversation_id):
    """Generates ATFs from conversational segments."""
    atfs = []
    for i, seg in enumerate(segments):
        logic_text = seg.strip()
        if not logic_text: continue
        
        concepts = extract_concepts(logic_text)
        my_id = f"{conversation_id}_{i}"
        
        # Action is based on the role/type
        role = "Assistant" if "assistant:" in logic_text.lower() else "User"
        action = f"Logic_{role}_{concepts[0].title()}" if concepts else f"Dialogue_{role}_{i}"
        
        # Connections (Edges)
        connections = [f"[{conversation_id}]"]
        connections.extend([f"[{c}]" for c in concepts])
        
        # Sanitize for Rust
        sanitized_logic = logic_text.replace('\\', '\\\\').replace('\"', '\\\"').replace('\n', ' ')
        
        atf = (
            f"## [ID: {my_id}]\n"
            f"**Action:** {action}\n"
            f"**Input:** {{Data}}\n"
            f"**Logic:** {sanitized_logic}\n"
            f"**Data_Connections:** {', '.join(connections)}\n"
            f"**Access:** Open\n"
            f"**Events:** Ingest\n\n"
        )
        atfs.append(atf)
    return "".join(atfs)

def run_beam_audit(limit=10):
    print("\n🚀 Initiating BEAM Forensic Audit (Mohammadta/BEAM 100K)...")
    try:
        ds = load_dataset("Mohammadta/BEAM", split="100K")
    except Exception as e:
        print(f"Error loading BEAM: {e}")
        return []

    results = []
    samples = list(ds)[:limit]
    
    for row in samples:
        conv_id = row.get("conversation_id", "unknown")
        chat = row.get("chat", [])
        
        # Flatten turns (Mocking AMB _iter_turns)
        turns = []
        for session in chat:
            if isinstance(session, list):
                for turn in session:
                    role = turn.get("role", "unknown").capitalize()
                    content = turn.get("content", "")
                    turns.append(f"{role}: {content}")
        
        if not turns: continue
        
        atf_markdown = generate_atfs(turns, conv_id)
        
        start_time = time.time()
        json_graph = fastmemory.process_markdown(atf_markdown)
        latency = (time.time() - start_time) * 1000
        
        cluster_count = json_graph.count('"block_type"')
        results.append({
            "Dataset": "BEAM-100K",
            "Sample_ID": conv_id,
            "Nodes": len(turns),
            "Clusters": cluster_count,
            "Latency_ms": latency
        })
        print(f"[BEAM] Processed {conv_id}: {len(turns)} turns -> {cluster_count} clusters in {latency:.2f}ms")
        
    return results

def run_personamem_audit(limit=10):
    print("\n🚀 Initiating PersonaMem Forensic Audit (bowen-upenn/PersonaMem)...")
    try:
        # PersonaMem contexts are in jsonl files in the hub
        local_path = hf_hub_download(repo_id="bowen-upenn/PersonaMem", filename="shared_contexts_32k.jsonl", repo_type="dataset")
        contexts = []
        with open(local_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                ctx_id, turns = next(iter(entry.items()))
                contexts.append((ctx_id, turns))
                if len(contexts) >= limit: break
    except Exception as e:
        print(f"Error loading PersonaMem: {e}")
        return []

    results = []
    for ctx_id, turns in contexts:
        segments = []
        for t in turns:
            role = t.get("role", "unknown")
            content = t.get("content", "")
            segments.append(f"[{role}] {content}")
            
        atf_markdown = generate_atfs(segments, ctx_id)
        
        start_time = time.time()
        json_graph = fastmemory.process_markdown(atf_markdown)
        latency = (time.time() - start_time) * 1000
        
        cluster_count = json_graph.count('"block_type"')
        results.append({
            "Dataset": "PersonaMem-32K",
            "Sample_ID": ctx_id,
            "Nodes": len(turns),
            "Clusters": cluster_count,
            "Latency_ms": latency
        })
        print(f"[PersonaMem] Processed {ctx_id}: {len(turns)} segments -> {cluster_count} clusters in {latency:.2f}ms")
        
    return results

def main():
    print("--- FASTMEMORY AUTHENTIC BEAM SOTA AUDIT ---")
    all_metrics = []
    
    # Run BEAM Audit (The primary correction)
    beam_results = run_beam_audit(limit=15)
    all_metrics.extend(beam_results)
    
    # Run PersonaMem Audit
    pm_results = run_personamem_audit(limit=10)
    all_metrics.extend(pm_results)
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv("authentic_fastmemory_metrics.csv", index=False)
        print("\n✅ CORRECTED BEAM AUDIT COMPLETE.")
        print("-" * 50)
        print(f"Total Logic Nodes: {df['Nodes'].sum()}")
        print(f"Avg Indexing Latency: {df['Latency_ms'].mean():.2f} ms")
        print(f"Total Topological Clusters: {df['Clusters'].sum()}")
        print("-" * 50)
        print("Final BEAM metrics saved to: authentic_fastmemory_metrics.csv")
    else:
        print("\n❌ Audit failed. Check logs.")

if __name__ == "__main__":
    main()
