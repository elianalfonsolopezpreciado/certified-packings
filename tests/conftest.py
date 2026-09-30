import os
import sys
from pathlib import Path

os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
