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
    print("!!! Critical Error: 'fastmemory' package not found.")
    print("Please run: pip install fastmemory==0.4.0")
    sys.exit(1)

def run_isolated_audit():
    print("--- [FORENSIC MODE] FastMemory SOTA Logic Audit ---")
    
    # 0. Engine Health Check
    print("[STEP 0] Checking Engine Binary Integrity...")
    test_atf = "## [ID: h]\n**Action:** A\n**Input:** {*}\n**Logic:** 1\n**Data_Connections:** [s]\n**Access:** O\n**Events:** N\n\n"
    try:
        res = fastmemory.process_markdown(test_atf)
        if res == "[]":
            print_critical_panic("Engine Health Check Failed: proprietary Louvain clustering logic failed to load.")
            return
        print("SUCCESS: Engine binary is responsive and clustering.")
    except Exception as e:
        print_critical_panic(f"Engine Load Crash: {e}")
        return

    # Enable Debug mode
    os.environ["FM_DEBUG"] = "1"
    
    # 1. Forensic ATF Payload (Example logic segments)
    docs = [
        Document("doc_company", "FastBuilder.AI is a leader in Sovereign AI.", "audit_user"),
        Document("doc_tech", "Our topological memory graphs achieve 100% SOTA on BEAM.", "audit_user"),
        Document("doc_login", "The master vault code is 1234-AX-99.", "audit_user")
    ]
    
    atf_blocks = []
    for doc in docs:
        sanit_content = doc.content.replace('\\', '\\\\').replace('\"', '\\\"')
        atf_blocks.append(
            f"## [ID: {doc.id}]\n"
            f"**Action:** Process_Logic\n"
            f"**Input:** {{Context}}\n"
            f"**Logic:** {sanit_content}\n"
            f"**Data_Connections:** [{doc.user_id}]\n"
            f"**Access:** Open\n"
            f"**Events:** Trigger_Audit\n\n"
        )
    atf_payload = "".join(atf_blocks)
    
    print("\n[STEP 1] Running Engine Indexing...")
    try:
        json_graph = fastmemory.process_markdown(atf_payload)
        if json_graph == "[]":
            print("FAILURE: Engine returned empty graph [].")
            return
        print(f"SUCCESS: Graph generated (len: {len(json_graph)})")
    except Exception as e:
        print(f"CRASH: Engine failed to process ATF: {e}")
        return

    # 2. Logic Recovery Check
    print("\n[STEP 2] Verifying Logic Retrieval...")
    if "1234-AX-99" in json_graph:
        print("SUCCESS: Logic '1234-AX-99' correctly linked in topological room.")
    else:
        print("FAILURE: Key logic not found in cluster.")

def print_critical_panic(detail):
    msg = f"""
################################################################################
#                                                                              #
#             !!! CRITICAL ENGINE FAILURE: FASTMEMORY PROPRIETARY !!!          #
#                                                                              #
################################################################################

FAILURE DETAIL: {detail}

DIAGNOSIS:
The topological clustering engine failed in this specific environment. 
This is a binary level conflict — likely an OS/Chipset mismatch for the 
compiled Rust core.

ACTION: 
1. Run `pip install --force-reinstall fastmemory==0.4.0`
2. Check if you are on an Intel Mac running Apple Silicon wheels (or vice-versa).
3. If issue persists, post your `uname -a` in the GitHub PR.

################################################################################
"""
    print(msg, file=sys.stderr)

if __name__ == "__main__":
    run_isolated_audit()
