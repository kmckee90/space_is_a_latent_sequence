"""
algorithm2.py
-------------
Algorithm 2: MAP Completion under a Rebound Emission  (Swaminathan et al. 2023)

After Algorithm 1 has produced a rebound emission vector, this algorithm
completes a prompt by:
  1. Running the forward pass on the prompt under the rebound emission to find
     the MAP (most probable) last hidden state.
  2. Greedily extending the sequence by always following the highest-probability
     transition until the delimiter token is emitted.

Usage
-----
    from rebind.model import CSCG
    from rebind.algorithm1 import rebind_fast
    from rebind.algorithm2 import complete_with_rebound

    emission_rb = rebind_fast(model, prompt_ids, ...)
    completion  = complete_with_rebound(model, prompt_ids, emission_rb,
                                        delimiter_token=delim_id)
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rebind.model import CSCG


def complete_with_rebound(
    model: "CSCG",
    x: np.ndarray,
    emission_rb: np.ndarray,
    delimiter_token: int,
    max_steps: int = 50,
) -> np.ndarray:
    """
    Greedy MAP completion given a rebound emission.

    Parameters
    ----------
    model           : trained CSCG
    x               : (T,) prompt token ids
    emission_rb     : (H,) rebound emission from rebind_fast()
    delimiter_token : token id that signals end-of-output
    max_steps       : safety cap on generation length

    Returns
    -------
    completion : 1-D array of generated token ids (excluding the prompt)
    """
    x = np.asarray(x, dtype=int)

    # Step 1: forward pass under rebound emission → MAP last hidden state
    log_alpha, _, _ = model.forward_backward_rebound(x, emission_rb)
    current_state = int(np.argmax(log_alpha[-1]))

    # Step 2: greedy rollout
    generated: List[int] = []
    for _ in range(max_steps):
        p, clone_i = _state_to_slot(model, current_state)

        # Find the next state with the highest single-step transition probability
        best_score = -np.inf
        best_next_state = -1

        for q in range(model.K):
            rq0 = int(model.token_offsets[q])
            row = model.A_tok[p][q][clone_i, :]   # (c_q,)
            local_best = int(np.argmax(row))
            score = float(row[local_best])
            if score > best_score:
                best_score = score
                best_next_state = rq0 + local_best

        if best_next_state < 0:
            break

        next_tok = int(emission_rb[best_next_state])
        if next_tok == delimiter_token:
            break

        generated.append(next_tok)
        current_state = best_next_state

    return np.array(generated, dtype=int)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _state_to_slot(model: "CSCG", h: int) -> Tuple[int, int]:
    """Return (token_k, clone_index) for global state h."""
    for k in range(model.K):
        r0 = int(model.token_offsets[k])
        r1 = int(model.token_offsets[k + 1])
        if r0 <= h < r1:
            return k, h - r0
    raise ValueError(f"State {h} out of range [0, {model.H})")
