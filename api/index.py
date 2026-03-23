import sys
from pathlib import Path

# Ensure src/ is importable when running as a Vercel serverless function
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from memory_bench.server import app  # noqa: F401  (Vercel picks up `app`)
