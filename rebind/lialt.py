"""
lialt.py
--------
LIALT: Language-Instructed Algorithm Learning Task  (Appendix D.1)

Defines the 13 algorithms, their instruction variants, and the data generator
that produces train/test demonstration sequences.

Each demonstration has the form:
    <instruction> / <input_1> <output_1> / ... / <input_n> <output_n> /

Vocabulary
----------
  VOCAB_TRAIN : 676 uppercase bigrams  (AA .. ZZ)  — training tokens
  VOCAB_TEST  : 676 lowercase bigrams  (aa .. zz)  — held-out test tokens
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List

# ---------------------------------------------------------------------------
# Dataset constants  (from Appendix D.1)
# ---------------------------------------------------------------------------

LIST_LEN_RANGE = (3, 6)     # list length drawn uniformly from this range (inclusive)
MATRIX_SIZES = [2, 3]       # matrix can be 2×2 or 3×3
N_ALGOS = 13                # number of algorithms used
N_INSTR_PER_ALGO = 10       # instruction variants per algorithm
N_DEMOS_PER_INSTR = 20      # demonstrations per (algorithm, instruction) pair
N_IO_PER_DEMO = 10          # input/output pairs per demonstration
DELIM = " / "               # delimiter between fields in a demonstration

VOCAB_TRAIN = [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
VOCAB_TEST  = [a + b for a in "abcdefghijklmnopqrstuvwxyz" for b in "abcdefghijklmnopqrstuvwxyz"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_list(xs: List[str]) -> str:
    return "[ " + " ".join(xs) + " ]"

def fmt_matrix(m: List[List[str]]) -> str:
    rows = ["[ " + " ".join(r) + " ]" for r in m]
    return "[ " + " ".join(rows) + " ]"


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_list(rng: random.Random, vocab: List[str]) -> List[str]:
    return [rng.choice(vocab) for _ in range(rng.randint(*LIST_LEN_RANGE))]

def sample_matrix(rng: random.Random, vocab: List[str]) -> List[List[str]]:
    n = rng.choice(MATRIX_SIZES)
    return [[rng.choice(vocab) for _ in range(n)] for _ in range(n)]


# ---------------------------------------------------------------------------
# The 13 algorithms
# ---------------------------------------------------------------------------

def list_idx0(xs: List) -> List:               return [xs[0]]
def list_idx1(xs: List) -> List:               return [xs[1]]
def list_idx2(xs: List) -> List:               return [xs[2]]
def list_reverse(xs: List) -> List:            return list(reversed(xs))
def list_duplicate(xs: List) -> List:          return [v for x in xs for v in (x, x)]
def list_circ_shift_fw(xs: List) -> List:      return [xs[-1]] + xs[:-1]   # right rotation
def list_circ_shift_bw(xs: List) -> List:      return xs[1:] + [xs[0]]     # left rotation
def list_alt_start_second(xs: List) -> List:   return xs[1::2]
def list_alt_start_first(xs: List) -> List:    return xs[0::2]

def matrix_diagonal(m: List[List]) -> List:
    return [m[i][i] for i in range(len(m))]

def matrix_transpose(m: List[List]) -> List[List]:
    n = len(m)
    return [[m[j][i] for j in range(n)] for i in range(n)]

def matrix_roll_columns_right(m: List[List]) -> List[List]:
    n = len(m)
    return [[m[i][(j - 1) % n] for j in range(n)] for i in range(n)]

def matrix_elem_row2_col2(m: List[List]) -> List:
    return [m[1][1]]    # second row, second column (1-indexed)


# ---------------------------------------------------------------------------
# Instruction variants  (Appendix D.1, Tables 2–3)
# ---------------------------------------------------------------------------

INSTRUCTIONS: dict[str, List[str]] = {
    "matrix_transpose": [
        "return the matrix transpose",
        "retrieve the transpose of the matrix",
        "get the transposed matrix",
        "compute the transposed form of the matrix",
        "derive the transpose matrix",
    ],
    "list_reverse": [
        "reverse the list",
        "mirror the list",
        "flip the list",
        "flip the order of the list",
        "reverse the order of the items in the list",
    ],
    "list_idx0": [
        "find the element at index zero of the list",
        "print the first element from the list",
        "return the leading element from the list",
        "find the head element from the list",
        "retrieve the starting element from the list",
    ],
    "list_idx1": [
        "print the element at index one of the list",
        "find the second element from the list",
        "retrieve the second element from the list",
        "locate the second item from the list",
        "return the element in second place from the list",
    ],
    "list_idx2": [
        "print the element at index two of the list",
        "find the third element from the list",
        "locate the third element from the list",
        "output the third item from the list",
        "return the element in third place from the list",
    ],
    "list_duplicate": [
        "duplicate each list item",
        "replicate every element in the list",
        "make a copy of each element in the list",
        "clone each element in the list",
        "create a second instance of every element in the list",
    ],
    "list_circ_shift_fw": [
        "rotate the list elements one place forward",
        "roll the list elements one position to the right",
        "switch the items of the list one position forward",
        "advance the list elements one index forward",
        "move the list elements one position forward",
    ],
    "list_alt_start_second": [
        "print every other member in the list starting with the second member",
        "retrieve alternate items in the list starting with the second item",
        "return every other object in the list starting with the second object",
        "retrieve every other entry in the list starting with the second entry",
        "output even indexed elements",
    ],
    "list_alt_start_first": [
        "print every other member in the list starting with the first member",
        "find alternate elements in the list beginning with the first element",
        "print every second item in the list, starting with the first element",
        "output every second element in the list, starting from the first element",
        "output odd indexed elements",
    ],
    "list_circ_shift_bw": [
        "rotate the list elements one place backward",
        "move the list elements one position to the left",
        "change the items of the list one position backward",
        "displace the elements of the list one index backward",
        "roll the list items one position backward",
    ],
    "matrix_diagonal": [
        "return the matrix diagonal",
        "collect the diagonal values of the matrix",
        "retrieve the diagonal elements of the matrix",
        "return the diagonal entries of the matrix",
        "fetch the diagonal items of the matrix",
    ],
    "matrix_roll_columns_right": [
        "roll the columns of the matrix to the right",
        "rotate the matrix columns to the right",
        "move the matrix columns to the right",
        "shift the columns of the matrix to the right",
        "spin the matrix columns to the right",
    ],
    "matrix_elem_row2_col2": [
        "find the matrix element in the second row and second column",
        "find the value in the second row and second column of the matrix",
        "fetch the matrix element located in row 2 and column 2",
        "print the value at 2 2 in the matrix",
        "retrieve the matrix element at 2 2",
    ],
}

ALGO_FUNCS: dict[str, Callable] = {
    "matrix_transpose":         matrix_transpose,
    "list_reverse":             list_reverse,
    "list_idx0":                list_idx0,
    "list_idx1":                list_idx1,
    "list_idx2":                list_idx2,
    "list_duplicate":           list_duplicate,
    "list_circ_shift_fw":       list_circ_shift_fw,
    "list_alt_start_second":    list_alt_start_second,
    "list_alt_start_first":     list_alt_start_first,
    "list_circ_shift_bw":       list_circ_shift_bw,
    "matrix_diagonal":          matrix_diagonal,
    "matrix_roll_columns_right": matrix_roll_columns_right,
    "matrix_elem_row2_col2":    matrix_elem_row2_col2,
}

# Trim to N_ALGOS and N_INSTR_PER_ALGO
INSTRUCTIONS = {k: v[:N_INSTR_PER_ALGO] for k, v in list(INSTRUCTIONS.items())[:N_ALGOS]}
ALGO_FUNCS   = dict(list(ALGO_FUNCS.items())[:N_ALGOS])

assert set(INSTRUCTIONS) == set(ALGO_FUNCS)
assert len(ALGO_FUNCS) == N_ALGOS

LIST_TASKS   = {k for k in ALGO_FUNCS if k.startswith("list_")}
MATRIX_TASKS = {k for k in ALGO_FUNCS if k.startswith("matrix_")}


# ---------------------------------------------------------------------------
# Demonstration builder
# ---------------------------------------------------------------------------

def make_demo(
    rng: random.Random,
    task_key: str,
    instruction: str,
    vocab: List[str],
) -> str:
    """Build one demonstration string for (task, instruction, vocab)."""
    parts = [instruction]
    for _ in range(N_IO_PER_DEMO):
        if task_key in LIST_TASKS:
            x = sample_list(rng, vocab)
            y = ALGO_FUNCS[task_key](x)
            parts.append(f"{fmt_list(x)} {fmt_list(y)}")
        else:
            x = sample_matrix(rng, vocab)
            y = ALGO_FUNCS[task_key](x)
            if task_key in {"matrix_transpose", "matrix_roll_columns_right"}:
                parts.append(f"{fmt_matrix(x)} {fmt_matrix(y)}")
            else:
                parts.append(f"{fmt_matrix(x)} {fmt_list(y)}")
    return DELIM.join(parts) + " /"


def generate_dataset(seed: int = 0, vocab: List[str] = VOCAB_TRAIN) -> str:
    """
    Generate the full LIALT dataset as a newline-separated string of
    demonstrations.  Each line is one demonstration.
    """
    rng = random.Random(seed)
    demos: List[str] = []
    for task_key, instrs in INSTRUCTIONS.items():
        for instruction in instrs:
            for _ in range(N_DEMOS_PER_INSTR):
                demos.append(make_demo(rng, task_key, instruction, vocab))
    return "\n".join(demos)


# ---------------------------------------------------------------------------
# CLI: write train/test files when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_text = generate_dataset(seed=123, vocab=VOCAB_TRAIN)
    with open("lialt_train.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    print(f"Wrote lialt_train.txt  ({len(train_text.splitlines())} demos)")

    test_text = generate_dataset(seed=123, vocab=VOCAB_TEST)
    with open("lialt_test.txt", "w", encoding="utf-8") as f:
        f.write(test_text)
    print(f"Wrote lialt_test.txt   ({len(test_text.splitlines())} demos)")
