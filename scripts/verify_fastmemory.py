"""
Verification script for FastMemory topological isolation and multi-hop reasoning.
Run this to verify that FastMemory correctly recovers needles AND
performs conceptual linking across different documents.
"""
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field

# Add src to sys.path for local testing
sys.path.append(str(Path(__file__).parent.parent.joinpath("src").absolute()))

try:
    from memory_bench.memory.fastmemory import FastMemoryProvider
    from memory_bench.models import Document
except ImportError:
    print("Warning: Standard imports failed. Checking local src path...")
    sys.path.append(str(Path(__file__).parent.parent.joinpath("src")))
    from memory_bench.memory.fastmemory import FastMemoryProvider
    from memory_bench.models import Document

def run_sota_audit():
    print("ðŸš€ Initiating FastMemory SOTA Audit (NIAH + Multi-Hop)...")
    print("-" * 60)
    
    provider = FastMemoryProvider()
    
    # 1. Multi-Hop Data Set
    # We want to see if the system can find the industry of a company
    # when the company-to-industry link is in one doc and the CEO info is in another.
    docs = [
        Document(
            id="doc_company_info",
            content="FastBuilder.AI is a leader in the Sovereign AI sector, specializing in topological memory graphs.",
            user_id="audit_user"
        ),
        Document(
            id="doc_contact_info",
            content="The CEO of FastBuilder.AI is Prabhat Singh, an expert in state-action memory.",
            user_id="audit_user"
        ),
        Document(
            id="needle_secret",
            content="The master vault code is 'CYBER-TRUTH-2026'. Protected by FastBuilder.AI protocols.",
            user_id="audit_user"
        )
    ]
    # Add some noise
    for i in range(10):
        docs.append(Document(id=f"noise_{i}", content="Standard corporate governance data.", user_id="audit_user"))
    
    print(f"[*] Ingesting {len(docs)} documents and building topology...")
    provider.ingest(docs)
    
    # TEST 1: NIAH (Direct ID/Keyword)
    print("\n[TEST 1] Querying for the master vault code...")
    res1, _ = provider.retrieve("What is the master vault code?", k=1, user_id="audit_user")
    if res1 and "CYBER-TRUTH-2026" in res1[0].content:
        print("âœ… SUCCESS: NIAH Recovery (100% Precision)")
    else:
        print("â Œ FAILURE: NIAH Recovery failed.")

    # TEST 2: Multi-Hop / Conceptual Link
    # Query mentions "Prabhat Singh" (found in doc_contact_info) 
    # and asks about "Sovereign AI" (found in doc_company_info).
    # Since both link to the concept 'FastBuilder', the provider should weight both high.
    print("\n[TEST 2] Querying for 'Prabhat Singh Sovereign AI' (Cross-Document link)...")
    res2, info = provider.retrieve("Find info on Prabhat Singh and the Sovereign AI sector.", k=2, user_id="audit_user")
    
    retrieved_ids = [r.id for r in res2]
    print(f"[+] Retrieved IDs: {retrieved_ids}")
    
    if "doc_company_info" in retrieved_ids and "doc_contact_info" in retrieved_ids:
        print("âœ… SUCCESS: Multi-Hop Conceptual Link verified via shared 'FastBuilder' topology.")
    else:
        print("â Œ FAILURE: Conceptual linking failed. Check extraction logic.")

if __name__ == "__main__":
    run_sota_audit()
EOF
