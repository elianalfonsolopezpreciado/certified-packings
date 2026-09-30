"""Thin wrapper: python scripts/export_pck.py certificates/A_n120.json "Author Name" [--shrink 1e-12] [--digits 20] > csqv120.pck

Implementation in search/export_pck.py (Packomania .pck layout per http://www.packomania.com/hints.html as read 2026-09-28: line 1 = largest
radius, line 2 = author(s), then `x y r` per circle sorted by increasing radius, coordinates centred at the container centre).
Nothing is ever sent by this project."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from search.export_pck import main  # noqa: E402

if __name__ == "__main__":
    main()
