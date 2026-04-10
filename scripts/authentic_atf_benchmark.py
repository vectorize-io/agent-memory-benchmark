import os
import time
import re
import string
import pandas as pd
from datasets import load_dataset
import fastmemory
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Ensure required NLTK packages are available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download issues (possible offline): {e}")

STOP_WORDS = {"this", "that", "these", "those", "when", "where", "which", "what", "there", "their", "after", "before", "will", "have", "with", "from"}

def extract_nouns(sentence):
    """Basic noun extraction fallback."""
    words = sentence.translate(str.maketrans('', '', string.punctuation)).split()
    return [w.lower() for w in words if len(w) > 4 and w.lower() not in STOP_WORDS]

def generate_atfs(sentences):
    """Generates complex ATFs with conceptual linkage across sentences."""
    atfs = []
    for i, s in enumerate(sentences):
        # We try to use NLTK for better entity extraction if available
        try:
            tokens = word_tokenize(s)
            tagged = pos_tag(tokens)
            nouns = [word.lower() for (word, pos) in tagged if pos.startswith('NN') and word.lower() not in STOP_WORDS]
        except:
            nouns = extract_nouns(s)
            
        my_id = f"REF_{i}"
        action = f"Logic_Process_{nouns[0].title()}" if nouns else f"Standard_Parse_{i}"
        connections = ", ".join([f"[{n}]" for n in nouns[:3]]) if nouns else "[Standard]"
        
        logic_content = s.replace('\\', '\\\\').replace('\"', '\\\"')
        atf = (
            f"## [ID: {my_id}]\n"
            f"**Action:** {action}\n"
            f"**Input:** {{Context}}\n"
            f"**Logic:** {logic_content}\n"
            f"**Data_Connections:** {connections}\n"
            f"**Access:** Open\n"
            f"**Events:** Trigger_Audit\n\n"
        )
        atfs.append(atf)
    return "".join(atfs)

def run_authentic_test(dataset_name, split, text_col, limit=20):
    print(f"\n🚀 Initiating Authentic Audit: {dataset_name} ({split})...")
    
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}. Skipping.")
        return None

    # Sample data
    samples = ds.select(range(min(limit, len(ds))))
    results = []

    for i, row in enumerate(samples):
        text = str(row.get(text_col, ""))
        if not text: continue
        
        # Split into logic segments
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s) > 10]
        if not sentences: continue
        
        atf_markdown = generate_atfs(sentences)
        
        start_time = time.time()
        try:
            json_graph = fastmemory.process_markdown(atf_markdown)
            latency = time.time() - start_time
            
            # Metric Derivation
            node_count = len(sentences)
            # Count clusters in JSON
            cluster_count = json_graph.count('"block_type"')
            
            results.append({
                "Sample_ID": i,
                "Nodes": node_count,
                "Clusters": cluster_count,
                "Latency_ms": latency * 1000,
                "Tokens": len(text.split()) * 1.3 # Rough approximation
            })
            print(f"[{dataset_name}] Processed sample {i}: {node_count} nodes -> {cluster_count} clusters in {latency*1000:.2f}ms")
            
        except Exception as e:
            print(f"FastMemory Error on sample {i}: {e}")

    return results

def main():
    print("--- FASTMEMORY AUTHENTIC REAL-WORLD BENCHMARK ---")
    
    all_metrics = []
    
    # Test 1: FinanceBench (Dense Financial Texts)
    fb_results = run_authentic_test("PatronusAI/financebench", "train", "evidence", limit=10)
    if fb_results: all_metrics.extend(fb_results)
    
    # Test 2: Google FRAMES (Multi-Doc Synthesis data - proxy)
    frames_results = run_authentic_test("google/frames-benchmark", "test", "Prompt", limit=10)
    if frames_results: all_metrics.extend(frames_results)
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv("authentic_fastmemory_metrics.csv", index=False)
        print("\n✅ AUTHENTIC AUDIT COMPLETE.")
        print("-" * 40)
        print(f"Total Logic Nodes Processed: {df['Nodes'].sum()}")
        print(f"Avg Indexing Latency: {df['Latency_ms'].mean():.2f} ms")
        print(f"Avg Clusters/Graph: {df['Clusters'].mean():.1f}")
        print("-" * 40)
        print("Detailed metrics saved to: authentic_fastmemory_metrics.csv")
    else:
        print("\n❌ Audit failed to produce metrics. Check dataset connectivity.")

if __name__ == "__main__":
    main()
