"""The locked end-to-end question file (ADR-0014). Only this module names it, and only
`grounded eval e2e --split locked --confirm-locked` reads it, once per model."""

from pathlib import Path

QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions" / "heldout.yaml"
