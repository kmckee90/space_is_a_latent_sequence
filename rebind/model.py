"""
model.py
--------
Clone Structured Causal Graph (CSCG) — a hidden Markov model variant where each
observable token is associated with a fixed group of latent "clone" states.

Key design:
  - Emissions are deterministic and fixed: clone group k emits token k.
  - The transition matrix A_tok[p][q] is a (c_p, c_q) block that is learned.
  - Training uses the EM algorithm (forward-backward) or Viterbi hard-assignment.

Public inference methods used by the rebinding algorithms:
  - forward_backward_rebound(x, emission)  → log_alpha, log_beta, loglik
  - leave_one_out_slot_probs(x, emission)  → (T, K) posterior over slots
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(amax), a - amax, 0.0)
    out = np.where(
        np.isfinite(amax),
        amax + np.log(np.sum(np.exp(safe), axis=axis, keepdims=True)),
        amax,
    )
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _row_normalize_blocks(blocks_row: List[np.ndarray]) -> List[np.ndarray]:
    """Row-normalize a list of matrices that together form one block-row of A."""
    if not blocks_row:
        return blocks_row
    row_sums = np.zeros((blocks_row[0].shape[0], 1), dtype=float)
    for B in blocks_row:
        row_sums += B.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    return [B / row_sums for B in blocks_row]


# ---------------------------------------------------------------------------
# CSCG model
# ---------------------------------------------------------------------------

class CSCG:
    """
    Clone Structured Causal Graph.

    Parameters
    ----------
    vocab_size : int
        Number of observable tokens K.
    clones_per_token : int or (K,) array
        Number of clone states per token.  All tokens share the same count when
        an int is provided.
    seed : int
        RNG seed for weight initialization.
    """

    def __init__(
        self,
        vocab_size: int,
        clones_per_token: Union[int, np.ndarray],
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.K = int(vocab_size)

        if np.isscalar(clones_per_token):
            c = int(clones_per_token)
            if c <= 0:
                raise ValueError("clones_per_token must be >= 1")
            self.clones_per_token = np.full(self.K, c, dtype=int)
        else:
            self.clones_per_token = np.asarray(clones_per_token, dtype=int)
            if self.clones_per_token.shape != (self.K,):
                raise ValueError("clones_per_token array must have shape (K,)")
            if np.any(self.clones_per_token <= 0):
                raise ValueError("all clones_per_token entries must be >= 1")

        # token_offsets[k] .. token_offsets[k+1] is the global-state slice for token k
        self.token_offsets = np.zeros(self.K + 1, dtype=int)
        self.token_offsets[1:] = np.cumsum(self.clones_per_token)
        self.H = int(self.token_offsets[-1])   # total number of hidden states

        # Initial clone distribution: pi_tok[k] has shape (c_k,)
        self.pi_tok: List[np.ndarray] = [
            np.full(int(self.clones_per_token[k]), 1.0 / int(self.clones_per_token[k]))
            for k in range(self.K)
        ]

        # Transition blocks: A_tok[p][q] is (c_p, c_q), row-normalized across all q
        self.A_tok: List[List[np.ndarray]] = []
        for p in range(self.K):
            cp = int(self.clones_per_token[p])
            row = [self.rng.random((cp, int(self.clones_per_token[q]))) for q in range(self.K)]
            self.A_tok.append(_row_normalize_blocks(row))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _logA_block(self, prev_tok: int, tok: int) -> np.ndarray:
        return np.log(np.maximum(self.A_tok[int(prev_tok)][int(tok)], 1e-300))

    def _logpi_vec(self, tok: int) -> np.ndarray:
        return np.log(np.maximum(self.pi_tok[int(tok)], 1e-300))

    # ------------------------------------------------------------------
    # Training: EM (forward-backward)
    # ------------------------------------------------------------------

    def _forward_backward(
        self, x: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """Standard (block-sparse) forward-backward for training sequences."""
        x = np.asarray(x, dtype=int)
        T = len(x)

        log_alpha: List[np.ndarray] = [None] * T  # type: ignore
        log_beta: List[np.ndarray] = [None] * T   # type: ignore

        log_alpha[0] = self._logpi_vec(int(x[0])).copy()
        for t in range(1, T):
            p, q = int(x[t - 1]), int(x[t])
            log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + self._logA_block(p, q), axis=0)

        loglik = float(_logsumexp(log_alpha[T - 1], axis=0))

        log_beta[T - 1] = np.zeros_like(log_alpha[T - 1])
        for t in range(T - 2, -1, -1):
            p, q = int(x[t]), int(x[t + 1])
            log_beta[t] = _logsumexp(self._logA_block(p, q) + log_beta[t + 1][None, :], axis=1)

        return log_alpha, log_beta, loglik

    def fit(
        self,
        sequences: Sequence[np.ndarray],
        n_iters: int = 100,
        pseudocount: float = 1e-6,
        verbose: bool = False,
    ) -> List[float]:
        """Baum-Welch EM training."""
        eps = float(pseudocount)
        logliks: List[float] = []

        for it in range(n_iters):
            exp_pi = [np.zeros_like(v) for v in self.pi_tok]
            exp_A: List[List[np.ndarray]] = [
                [np.zeros((int(self.clones_per_token[p]), int(self.clones_per_token[q])))
                 for q in range(self.K)]
                for p in range(self.K)
            ]
            total_ll = 0.0

            for x in sequences:
                x = np.asarray(x, dtype=int)
                T = len(x)
                log_alpha, log_beta, ll = self._forward_backward(x)
                total_ll += ll

                gamma = [np.exp(log_alpha[t] + log_beta[t] - ll) for t in range(T)]
                exp_pi[int(x[0])] += gamma[0]

                for t in range(T - 1):
                    p, q = int(x[t]), int(x[t + 1])
                    exp_A[p][q] += np.exp(
                        log_alpha[t][:, None] + self._logA_block(p, q) + log_beta[t + 1][None, :] - ll
                    )

            for k in range(self.K):
                v = exp_pi[k] + eps
                self.pi_tok[k] = v / v.sum()

            for p in range(self.K):
                self.A_tok[p] = _row_normalize_blocks([exp_A[p][q] + eps for q in range(self.K)])

            logliks.append(total_ll)
            if verbose:
                print(f"EM iter {it + 1}/{n_iters}  loglik={total_ll:.6f}")

        return logliks

    # ------------------------------------------------------------------
    # Training: Viterbi hard-assignment
    # ------------------------------------------------------------------

    def viterbi(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Viterbi decoding; returns global state path and log-probability."""
        x = np.asarray(x, dtype=int)
        T = len(x)

        dp = [None] * T
        back = [None] * T

        dp[0] = self._logpi_vec(int(x[0])).copy()
        back[0] = np.full(dp[0].shape, -1, dtype=int)

        for t in range(1, T):
            p, q = int(x[t - 1]), int(x[t])
            scores = dp[t - 1][:, None] + self._logA_block(p, q)
            back[t] = np.argmax(scores, axis=0)
            dp[t] = np.max(scores, axis=0)

        best_last = int(np.argmax(dp[T - 1]))
        best_ll = float(dp[T - 1][best_last])

        clone_path = np.empty(T, dtype=int)
        clone_path[T - 1] = best_last
        for t in range(T - 2, -1, -1):
            clone_path[t] = int(back[t + 1][clone_path[t + 1]])

        state_path = np.empty(T, dtype=int)
        for t in range(T):
            state_path[t] = int(self.token_offsets[int(x[t])]) + clone_path[t]

        return state_path, best_ll

    def viterbi_train(
        self,
        sequences: Sequence[np.ndarray],
        n_iters: int = 10,
        pseudocount: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        """Hard-assignment Viterbi training."""
        eps = float(pseudocount)

        for it in range(n_iters):
            exp_pi = [np.zeros_like(v) for v in self.pi_tok]
            exp_A: List[List[np.ndarray]] = [
                [np.zeros((int(self.clones_per_token[p]), int(self.clones_per_token[q])))
                 for q in range(self.K)]
                for p in range(self.K)
            ]
            total_ll = 0.0

            for x in sequences:
                x = np.asarray(x, dtype=int)
                state_path, ll = self.viterbi(x)
                total_ll += ll

                def clone_idx(tok: int, state: int) -> int:
                    return int(state) - int(self.token_offsets[int(tok)])

                exp_pi[int(x[0])][clone_idx(int(x[0]), state_path[0])] += 1.0
                for t in range(len(x) - 1):
                    p, q = int(x[t]), int(x[t + 1])
                    exp_A[p][q][clone_idx(p, state_path[t]), clone_idx(q, state_path[t + 1])] += 1.0

            for k in range(self.K):
                v = exp_pi[k] + eps
                self.pi_tok[k] = v / v.sum()

            for p in range(self.K):
                self.A_tok[p] = _row_normalize_blocks([exp_A[p][q] + eps for q in range(self.K)])

            if verbose:
                print(f"Viterbi iter {it + 1}/{n_iters}  path-loglik-sum={total_ll:.6f}")

    # ------------------------------------------------------------------
    # Inference primitives used by the rebinding algorithms
    # ------------------------------------------------------------------

    def forward_backward_rebound(
        self,
        x: np.ndarray,
        emission: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Forward-backward over all H global states under a (possibly rebound) emission.

        Novel tokens (no state emits them) are treated as missing observations:
        their emission factor is uniform (1 for all states) so the transition
        structure can still propagate context across those positions.

        Parameters
        ----------
        x        : (T,) token ids
        emission : (H,) array — emission[h] = token currently emitted by state h

        Returns
        -------
        log_alpha, log_beta : each a T-list of (H,) arrays
        loglik : float
        """
        x = np.asarray(x, dtype=int)
        T = len(x)

        def log_obs_vec(tok: int) -> np.ndarray:
            mask = emission == tok
            if mask.any():
                v = np.full(self.H, -np.inf)
                v[mask] = 0.0
                return v
            return np.zeros(self.H)  # novel token → uniform (no information)

        def fwd_step(la_prev: np.ndarray) -> np.ndarray:
            la = np.full(self.H, -np.inf)
            for p in range(self.K):
                r0, r1 = int(self.token_offsets[p]), int(self.token_offsets[p + 1])
                ap = la_prev[r0:r1]
                if np.all(ap == -np.inf):
                    continue
                for q in range(self.K):
                    c0, c1 = int(self.token_offsets[q]), int(self.token_offsets[q + 1])
                    contrib = _logsumexp(ap[:, None] + self._logA_block(p, q), axis=0)
                    la[c0:c1] = _logsumexp(np.stack([la[c0:c1], contrib]), axis=0)
            return la

        def bwd_step(lb_next_obs: np.ndarray) -> np.ndarray:
            lb = np.full(self.H, -np.inf)
            for p in range(self.K):
                r0, r1 = int(self.token_offsets[p]), int(self.token_offsets[p + 1])
                for q in range(self.K):
                    c0, c1 = int(self.token_offsets[q]), int(self.token_offsets[q + 1])
                    contrib = _logsumexp(self._logA_block(p, q) + lb_next_obs[c0:c1][None, :], axis=1)
                    lb[r0:r1] = _logsumexp(np.stack([lb[r0:r1], contrib]), axis=0)
            return lb

        # Forward
        log_alpha: List[np.ndarray] = [None] * T  # type: ignore
        tok0 = int(x[0])
        la0 = np.full(self.H, -np.inf)
        if (emission == tok0).any():
            r0, r1 = int(self.token_offsets[tok0]), int(self.token_offsets[tok0 + 1])
            la0[r0:r1] = self._logpi_vec(tok0)
        else:
            la0[:] = -np.log(self.H)
        la0 += log_obs_vec(tok0)
        log_alpha[0] = la0

        for t in range(1, T):
            log_alpha[t] = fwd_step(log_alpha[t - 1]) + log_obs_vec(int(x[t]))

        loglik = float(_logsumexp(log_alpha[T - 1], axis=0))

        # Backward
        log_beta: List[np.ndarray] = [None] * T  # type: ignore
        log_beta[T - 1] = np.zeros(self.H)
        for t in range(T - 2, -1, -1):
            log_beta[t] = bwd_step(log_beta[t + 1] + log_obs_vec(int(x[t + 1])))

        return log_alpha, log_beta, loglik

    def leave_one_out_slot_probs(
        self,
        x: np.ndarray,
        emission: np.ndarray,
    ) -> np.ndarray:
        """
        p(slot_n = j | x_{-n}) for all positions n and schema slots j.

        This is the bidirectional leave-one-out posterior used in Algorithm 1
        (Steps 3 & 4) to identify anchor positions and rebinding candidates.
        A schema slot j = clone group {token_offsets[j], ..., token_offsets[j+1)-1}.

        Returns
        -------
        loo : (T, K) array  — loo[n, j] = p(slot_n = j | x_{-n})
        """
        x = np.asarray(x, dtype=int)
        T = len(x)
        log_alpha, log_beta, _ = self.forward_backward_rebound(x, emission)

        def log_obs_vec(tok: int) -> np.ndarray:
            mask = emission == tok
            if mask.any():
                v = np.full(self.H, -np.inf)
                v[mask] = 0.0
                return v
            return np.zeros(self.H)

        def fwd_step_no_obs(la_prev: np.ndarray) -> np.ndarray:
            la = np.full(self.H, -np.inf)
            for p in range(self.K):
                r0, r1 = int(self.token_offsets[p]), int(self.token_offsets[p + 1])
                ap = la_prev[r0:r1]
                if np.all(ap == -np.inf):
                    continue
                for q in range(self.K):
                    c0, c1 = int(self.token_offsets[q]), int(self.token_offsets[q + 1])
                    contrib = _logsumexp(ap[:, None] + self._logA_block(p, q), axis=0)
                    la[c0:c1] = _logsumexp(np.stack([la[c0:c1], contrib]), axis=0)
            return la

        # pred_alpha[n] = T * alpha[n-1] (no emission applied at n)
        pred_alpha: List[np.ndarray] = [None] * T  # type: ignore
        tok0 = int(x[0])
        pa0 = np.full(self.H, -np.inf)
        if tok0 < self.K and (emission == tok0).any():
            r0, r1 = int(self.token_offsets[tok0]), int(self.token_offsets[tok0 + 1])
            pa0[r0:r1] = self._logpi_vec(tok0)
        else:
            pa0[:] = -np.log(self.H)
        pred_alpha[0] = pa0

        for t in range(1, T):
            pred_alpha[t] = fwd_step_no_obs(log_alpha[t - 1])

        loo = np.zeros((T, self.K))
        for n in range(T):
            log_joint = pred_alpha[n] + log_beta[n]
            log_Z = _logsumexp(log_joint, axis=0)
            for j in range(self.K):
                r0, r1 = int(self.token_offsets[j]), int(self.token_offsets[j + 1])
                loo[n, j] = float(np.exp(_logsumexp(log_joint[r0:r1], axis=0) - log_Z))

        return loo  # (T, K)

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------

    def reduced_transition_matrix(self, threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble the full H×H matrix and return the submatrix restricted to
        states that have at least one edge above `threshold`.

        Returns (A_reduced, active_state_indices).
        """
        A_full = np.zeros((self.H, self.H))
        for p in range(self.K):
            r0, r1 = int(self.token_offsets[p]), int(self.token_offsets[p + 1])
            for q in range(self.K):
                c0, c1 = int(self.token_offsets[q]), int(self.token_offsets[q + 1])
                A_full[r0:r1, c0:c1] = self.A_tok[p][q]

        active_mask = np.any(A_full > threshold, axis=1) | np.any(A_full > threshold, axis=0)
        active_states = np.where(active_mask)[0]
        return A_full[np.ix_(active_states, active_states)], active_states
