"""
data.py
-------
Tokenization, vocabulary, and test-file parsing for the LIALT experiment.

The LIALT sequences use a simple token grammar:
  - "[" and "]"  : list/matrix delimiters
  - "/"          : field separator (also the end-of-sequence marker)
  - word tokens  : instruction words and data tokens (bigrams like AB, xy ...)

Vocab split
-----------
  train_vocab  : tokens seen in training → determines model size K
  full_vocab   : train + test-only tokens

Test-only tokens (ids >= K) never appear in the base emission, so the rebinding
forward-backward treats them as novel and lets the transition structure identify
the correct schema slot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_TOK_PATTERN = re.compile(r"\[|\]|/|[A-Za-z0-9]+")

def tokenize(s: str) -> List[str]:
    """Split a raw string into the LIALT token list."""
    return _TOK_PATTERN.findall(s)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(token_sequences: List[List[str]]) -> Dict[str, int]:
    """
    Assign a unique integer id to every token type that appears in the
    provided sequences, in first-encounter order.
    """
    vocab: Dict[str, int] = {}
    for seq in token_sequences:
        for tok in seq:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode(seq: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """Map a token list to an integer id array using vocab."""
    return np.asarray([vocab[tok] for tok in seq], dtype=np.int32)


# ---------------------------------------------------------------------------
# Prompt dataclass
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    """A single evaluation prompt."""
    prompt_tokens: List[str]   # everything before the expected completion
    target_tokens: List[str]   # the expected completion (including trailing "/")
    task: str = ""             # optional label for display


# ---------------------------------------------------------------------------
# Test file parsing
# ---------------------------------------------------------------------------

def parse_test_file(path: str, n: int) -> Tuple[List[Prompt], List[Prompt]]:
    """
    Parse a LIALT test file into two lists of Prompts.

    Each line in the file is a complete demonstration, e.g.:
        reverse the list / [ AB CD EF ] [ EF CD AB ] / [ GH IJ ] [ IJ GH ] /

    From each line we extract two prompts:
      - Instruction-based  : "instruction / last_input"  →  predict "last_output /"
      - Example-based      : "prev_input prev_output / last_input"  →  predict "last_output /"

    Parameters
    ----------
    path : str   path to the test text file
    n    : int   maximum number of prompts to return

    Returns
    -------
    (instr_prompts, example_prompts)
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    def split_input_output(io_field: str) -> Tuple[str, str]:
        """Split '[ a b ] [ c d ]' into its input part and output part."""
        toks = tokenize(io_field)
        depth = 0
        for i, t in enumerate(toks):
            if t == "[":
                depth += 1
            elif t == "]":
                depth -= 1
                if depth == 0:
                    if i + 1 >= len(toks):
                        raise ValueError(f"No output part in: {io_field!r}")
                    return " ".join(toks[:i + 1]), " ".join(toks[i + 1:])
        raise ValueError(f"Unbalanced brackets in: {io_field!r}")

    instr_prompts: List[Prompt] = []
    example_prompts: List[Prompt] = []

    for line in lines:
        fields = [f.strip() for f in line.split(" / ") if f.strip()]
        if len(fields) < 3:
            continue   # need instruction + at least two io pairs

        instr = fields[0]
        io_fields = fields[1:]

        try:
            last_in, last_out   = split_input_output(io_fields[-1])
            prev_in,  prev_out  = split_input_output(io_fields[-2])
        except ValueError:
            continue

        # Instruction-based prompt: instruction + last input → last output
        instr_prompts.append(Prompt(
            prompt_tokens=tokenize(f"{instr} / {last_in}"),
            target_tokens=tokenize(f" {last_out} /"),
            task=instr,
        ))

        # Example-based prompt: one in/out example + last input → last output
        example_prompts.append(Prompt(
            prompt_tokens=tokenize(f"{prev_in} {prev_out} / {last_in}"),
            target_tokens=tokenize(f" {last_out} /"),
            task="(example)",
        ))

        if len(instr_prompts) >= n:
            break

    return instr_prompts, example_prompts
