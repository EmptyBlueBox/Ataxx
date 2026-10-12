import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
os.execv(root / "build/engine", ["engine", str(root / "weights/champion.onnx"), "botzone", "11.1", "3.7", str(root / "book.txt")])
