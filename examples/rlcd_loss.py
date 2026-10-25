"""Toy demo of the standard RLCD loss; the implementation lives in decision/losses.py.

Run from the repository root:  python examples/rlcd_loss.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision.losses import demo  # noqa: E402

if __name__ == "__main__":
    demo()
