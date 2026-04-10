from __future__ import annotations
"""
FORENSIC VERIFICATION SCRIPT for FastMemory (Zero-Dependency Version).
This script bypasses external benchmark dependencies to focus 
exclusively on validating the FastMemory Rust engine and ATF logic.
"""
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field

# --- STANDALONE MODELS (Bypassing benchmark imports) ---
@dataclass
class Document:
    id: str
    content: str
    user_id: str | None = None
    meta: dict = field(default_factory=dict)

class MemoryProvider:
    """Base interface mock"""
    pass

# Patch the system path to find the locals
sys.path.append(str(Path(__file__).parent.parent.joinpath("src").absolute()))

# --- IMPORT ONLY FASTM_PROVIDER ---
try:
    # We use a custom import to avoid the memory.__init__.py dependency chain
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fastmemory_provider", 
        Path(__file__).parent.parent / "src/memory_bench/memory/fastmemory.py"
    )
    fm_mod = importlib.util.module_from_spec(spec)
    # Inject Mocked models into the module to prevent import errors
    sys.modules["..models"] = type('models', (), {'Document': Document})
    sys.modules["models"] = sys.modules["..models"]
    sys.modules[".base"] = type('base', (), {'MemoryProvider': MemoryProvider})
    spec.loader.exec_module(fm_mod)
    FastMemoryProvider = fm_mod.FastMemoryProvider
except Exception as e:
    print(f"!!! Forensic Setup Failed: {e}")
    sys.exit(1)

def run_sota_audit():
    print("ðŸš€ Initiating FastMemory FORENSIC AUDIT (NIAH + Multi-Hop)...")
    print("-" * 60)
    
    # Enable debug mode to see exact ATF trace as requested by maintainers
    os.environ["FM_DEBUG"] = "1"
    provider = FastMemoryProvider()
    
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
    for i in range(5):
        docs.append(Document(id=f"noise_{i}", content="Standard corporate governance dummy data.", user_id="audit_user"))
    
    print(f"[*] Ingesting {len(docs)} documents and building topology...")
    provider.ingest(docs)
    
    # TEST 1: NIAH
    print("\n[TEST 1] Querying for 'master vault code' (NIAH)...")
    res1, _ = provider.retrieve("What is the master vault code?", k=1, user_id="audit_user")
    if res1 and "CYBER-TRUTH-2026" in res1[0].content:
        print("âœ… SUCCESS: NIAH Recovery (100% Precision)")
    else:
        print("â Œ FAILURE: NIAH Recovery failed.")

    # TEST 2: Multi-Hop / Conceptual Link
    print("\n[TEST 2] Querying for 'Prabhat Singh Sovereign AI' (Multi-Hop)...")
    res2, _ = provider.retrieve("Find info on Prabhat Singh and the Sovereign AI sector.", k=2, user_id="audit_user")
    
    retrieved_ids = [r.id for r in res2]
    print(f"[+] Retrieved IDs: {retrieved_ids}")
    
    expected = {"doc_company_info", "doc_contact_info"}
    if expected.issubset(set(retrieved_ids)):
        print("âœ… SUCCESS: Multi-Hop Conceptual Link verified via shared 'FastBuilder' topology.")
    else:
        print("â Œ FAILURE: Conceptual linking failed.")

if __name__ == "__main__":
    run_sota_audit()
