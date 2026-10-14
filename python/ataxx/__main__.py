import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
os.execv(root / "build/engine", ["engine", str(root / "weights/champion.bin"), "botzone", "11.6", "3.85"])
