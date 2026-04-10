"""
eval.py
-------
Evaluation harness for the rebinding experiment.

For each prompt:
  1. Run Algorithm 1 (rebind_fast) on the prompt prefix.
  2. Run Algorithm 2 (complete_with_rebound) to generate a completion.
  3. Compare the generated tokens to the ground-truth target.
  4. Report exact-match and token-level accuracy.
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import numpy as np

from rebind.algorithm1 import rebind_fast
from rebind.algorithm2 import complete_with_rebound
from rebind.data import Prompt, encode

if TYPE_CHECKING:
    from rebind.model import CSCG


def eval_with_rebinding(
    model: "CSCG",
    prompts: List[Prompt],
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    delimiter_id: int,
    eps: float = 1e-6,
    p_surprise: float = 0.1,
    n_iters: int = 1,
    label: str = "",
    show_every: int = 10,
) -> float:
    """
    Evaluate a list of prompts using rebinding + completion.

    Parameters
    ----------
    model        : trained CSCG
    prompts      : list of Prompt objects
    vocab        : token → id mapping (includes test-only tokens)
    inv_vocab    : id → token mapping
    delimiter_id : token id for "/"  (end-of-output signal)
    eps          : smoothing passed to rebind_fast
    p_surprise   : confidence threshold θ for Algorithm 1
    n_iters      : Algorithm 1 EM iterations per prompt
    label        : display name for this evaluation split
    show_every   : print rollout details for every N-th prompt (0 = silent)

    Returns
    -------
    exact_accuracy : fraction of prompts with exact-match completion
    """
    n_exact = 0
    n_tok_correct = 0
    n_tok_total = 0
    n_evaluated = 0
    n_skipped = 0

    print(f"\n{'=' * 60}")
    print(f"  Eval: {label}  ({len(prompts)} prompts)")
    print(f"  eps={eps}  p_surprise={p_surprise}  n_iters={n_iters}")
    print(f"{'=' * 60}")

    for idx, p in enumerate(prompts):
        all_toks = p.prompt_tokens + p.target_tokens
        if any(t not in vocab for t in all_toks):
            n_skipped += 1
            continue

        prompt_ids = encode(p.prompt_tokens, vocab)
        target_core = [t for t in p.target_tokens if t != "/"]

        # Algorithm 1: rebind the emission to the prompt context
        emission_rb = rebind_fast(
            model, prompt_ids, eps=eps, p_surprise=p_surprise, n_iters=n_iters
        )

        # Algorithm 2: complete under the rebound emission
        gen_ids = complete_with_rebound(
            model, prompt_ids, emission_rb,
            delimiter_token=delimiter_id,
            max_steps=len(p.target_tokens) + 5,
        )
        gen_toks = [inv_vocab[i] for i in gen_ids]
        gen_core = [t for t in gen_toks if t != "/"]

        # Exact-match accuracy
        is_exact = gen_core == target_core
        if is_exact:
            n_exact += 1

        # Token-level accuracy (penalise short generations)
        for pred_t, true_t in zip(gen_toks, p.target_tokens):
            n_tok_correct += int(pred_t == true_t)
            n_tok_total += 1
        n_tok_total += max(0, len(p.target_tokens) - len(gen_toks))

        n_evaluated += 1

        # Diagnostic printout
        if show_every > 0 and idx % show_every == 0:
            rebound_pairs = _rebound_summary(model, emission_rb, inv_vocab)
            status = "✓" if is_exact else "✗"
            print(f"\n[{idx:3d}] {status}  task: {p.task}")
            print(f"  prompt   : {' '.join(p.prompt_tokens)}")
            print(f"  target   : {' '.join(p.target_tokens)}")
            print(f"  generated: {' '.join(gen_toks) if gen_toks else '(empty)'}")
            print(f"  rebound  : {rebound_pairs if rebound_pairs else '(none)'}")

    exact_acc = n_exact / max(1, n_evaluated)
    tok_acc   = n_tok_correct / max(1, n_tok_total)

    print(f"\n--- {label} results ---")
    print(f"  Evaluated : {n_evaluated}  (skipped {n_skipped} OOV)")
    print(f"  Exact acc : {exact_acc:.3f}  ({n_exact}/{n_evaluated})")
    print(f"  Token acc : {tok_acc:.3f}  ({n_tok_correct}/{n_tok_total})")

    return exact_acc


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _rebound_summary(
    model: "CSCG",
    emission_rb: np.ndarray,
    inv_vocab: Dict[int, str],
) -> list:
    """Return a sorted list of (original_token, rebound_token) pairs for slots
    whose emission changed from the identity baseline."""
    base = np.empty(model.H, dtype=int)
    for k in range(model.K):
        r0, r1 = int(model.token_offsets[k]), int(model.token_offsets[k + 1])
        base[r0:r1] = k
    changed = np.where(emission_rb != base)[0]
    return sorted({
        (inv_vocab.get(int(base[h]), str(base[h])),
         inv_vocab.get(int(emission_rb[h]), str(emission_rb[h])))
        for h in changed
    })
