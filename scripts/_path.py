"""Make `occwm` importable when running scripts directly (python scripts/foo.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
