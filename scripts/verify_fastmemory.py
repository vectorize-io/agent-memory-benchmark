import os
import sys
import json
import time

# ZERO DEPENDENCY MOCK MODELS
class Document:
    def __init__(self, id, content, user_id):
        self.id = id
        self.content = content
        self.user_id = user_id

class Query:
    def __init__(self, query):
        self.query = query

try:
    import fastmemory
except ImportError:
    print("!!! Error: 'fastmemory' package not found.")
    print("Please run: pip install fastmemory>=0.4.3")
    sys.exit(1)

def run_isolated_audit():
    print("--- [FORENSIC MODE] FastMemory Engine Audit ---")
    
    # 0. Engine Health Check
    print("[STEP 0] Checking Engine Health...")
    test_input = "The quick brown fox jumps over the lazy dog. Cats are independent animals."
    try:
        res = fastmemory.process_markdown(test_input)
        if res == "[]":
            print("FAILURE: Engine returned empty graph.")
            print("DIAGNOSIS: The embedded rust-louvain binary may not be compatible with your platform.")
            print(f"  Platform: {sys.platform}, Python: {sys.version}")
            print("ACTION: pip install --force-reinstall fastmemory>=0.4.3")
            return
        print(f"SUCCESS: Engine is responsive (output: {len(res)} chars)")
    except Exception as e:
        print(f"CRASH: Engine failed: {e}")
        return

    # 1. Forensic Payload
    docs = [
        Document("doc_company", "FastBuilder.AI is a leader in Sovereign AI.", "audit_user"),
        Document("doc_tech", "Our topological memory graphs achieve high precision on BEAM.", "audit_user"),
        Document("doc_login", "The master vault code is 1234-AX-99.", "audit_user")
    ]
    
    segments = [doc.content for doc in docs]
    full_text = " ".join(segments)
    
    print("\n[STEP 1] Running Engine Indexing...")
    try:
        json_graph = fastmemory.process_markdown(full_text)
        if json_graph == "[]":
            print("FAILURE: Engine returned empty graph [].")
            return
        print(f"SUCCESS: Graph generated (len: {len(json_graph)})")
    except Exception as e:
        print(f"CRASH: Engine failed to process input: {e}")
        return

    # 2. Content Recovery Check
    print("\n[STEP 2] Verifying Topology Structure...")
    try:
        graph = json.loads(json_graph)
        total_nodes = sum(len(block.get("nodes", [])) for block in graph)
        print(f"SUCCESS: {len(graph)} clusters, {total_nodes} total nodes")
    except json.JSONDecodeError:
        print("FAILURE: Engine returned invalid JSON")

if __name__ == "__main__":
    run_isolated_audit()
