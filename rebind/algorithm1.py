"""
algorithm1.py
-------------
Algorithm 1: Fast Rebinding  (Swaminathan et al. 2023, Algorithm 1)

Given a prompt sequence x (which may contain novel tokens unseen at training
time), identify which schema slots the model would predict at each position and
rebind those slots to emit the tokens actually observed.

The key steps are:
  1. Start from the base emission (slot j emits token j).
  2. Run bidirectional leave-one-out inference to get p(slot_n = j | x_{-n}).
  3. Mark positions/slots as **anchors** when the prediction is confident and
     already correct — leave those alone.
  4. Accumulate affinity scores for (slot, target-token) pairs at all remaining
     surprised positions.
  5. Solve a one-to-one (injective) linear assignment to rebind each candidate
     slot to its best target token.
  6. Repeat for n_iters EM iterations.

Usage
-----
    from rebind.model import CSCG
    from rebind.algorithm1 import rebind_fast

    emission_rb = rebind_fast(model, prompt_ids, eps=1e-6, p_surprise=0.1, n_iters=1)
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from rebind.model import CSCG


def rebind_fast(
    model: "CSCG",
    x: np.ndarray,
    eps: float = 1e-6,
    p_surprise: float = 0.1,
    n_iters: int = 1,
) -> np.ndarray:
    """
    Fast rebinding (Algorithm 1).

    Parameters
    ----------
    model      : trained CSCG
    x          : (T,) prompt token ids (may include novel tokens >= model.K)
    eps        : additive smoothing (unused in current implementation but
                 matches the pseudocount spirit of the paper)
    p_surprise : confidence threshold θ — a slot must exceed this posterior
                 to be considered as anchor or rebinding candidate
    n_iters    : number of EM iterations over the prompt

    Returns
    -------
    emission_rb : (H,) array — emission_rb[h] = token emitted by state h after
                  rebinding.  Starts as the identity (slot j → token j) and is
                  updated in-place over iterations.
    """
    x = np.asarray(x, dtype=int)

    # Step 1: base emission — slot j emits token j (identity mapping)
    emission = np.empty(model.H, dtype=int)
    for k in range(model.K):
        r0, r1 = int(model.token_offsets[k]), int(model.token_offsets[k + 1])
        emission[r0:r1] = k

    for _iter in range(n_iters):

        # Step 2: bidirectional leave-one-out slot posteriors  shape (T, K)
        loo = model.leave_one_out_slot_probs(x, emission)

        # ------------------------------------------------------------------
        # Step 3: Anchors
        #   Position n is anchored when some slot j is predicted with
        #   confidence > p_surprise AND slot j already emits the observed
        #   token x_n.  These positions are correct; do not rebind them.
        # ------------------------------------------------------------------
        anchor_positions: set = set()
        anchor_slots: set = set()

        for n in range(len(x)):
            tok_n = int(x[n])
            for j in range(model.K):
                if loo[n, j] > p_surprise:
                    current_tok = int(emission[int(model.token_offsets[j])])
                    if current_tok == tok_n:
                        anchor_positions.add(n)
                        anchor_slots.add(j)

        # ------------------------------------------------------------------
        # Step 4: Rebinding candidates
        #   Accumulate affinity(slot j → target token) over all non-anchor
        #   positions where slot j is confidently predicted but currently
        #   emits the wrong token.
        # ------------------------------------------------------------------
        affinity: Dict[int, Dict[int, float]] = {}   # slot_j → {target_tok: score}

        for n in range(len(x)):
            if n in anchor_positions:
                continue
            tok_n = int(x[n])
            for j in range(model.K):
                if j in anchor_slots:
                    continue
                if loo[n, j] <= p_surprise:
                    continue
                if int(emission[int(model.token_offsets[j])]) == tok_n:
                    continue  # already correct
                affinity.setdefault(j, {})[tok_n] = affinity.get(j, {}).get(tok_n, 0.0) + float(loo[n, j])

        if not affinity:
            break  # nothing to rebind

        # ------------------------------------------------------------------
        # Step 5: Injective slot ↔ target assignment  (linear sum assignment)
        #   All clones within a slot receive the same new target token.
        # ------------------------------------------------------------------
        slots = sorted(affinity)
        targets = sorted({t for d in affinity.values() for t in d})

        affinity_mat = np.zeros((len(slots), len(targets)))
        for i, j in enumerate(slots):
            for t, score in affinity[j].items():
                affinity_mat[i, targets.index(t)] += score

        row_ind, col_ind = linear_sum_assignment(-affinity_mat)   # maximise

        for i, ci in zip(row_ind, col_ind):
            if affinity_mat[i, ci] > 0:
                j = slots[i]
                tgt = targets[ci]
                r0, r1 = int(model.token_offsets[j]), int(model.token_offsets[j + 1])
                emission[r0:r1] = tgt

    return emission
