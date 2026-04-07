# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CSCG-HE split EM v2: split transition-counts accumulation + scatter-add elsewhere.

Extends cscg_he_split_em (which already splits the EM sum-product forward and
backward passes) by also splitting and parallelising the remaining sequential
operations that were bottlenecking overall speedup.

New rebinds (all relative to cscg_he_split_em / cscg_he):

  _update_transition_counts    -- split into num_splits parallel segment scans
                                   (same segment granularity as the EM passes)
  _update_emission_counts      -- scatter-add over T time steps (no scan)
  _update_transition_counts_mp -- scatter-add (replaces sequential scan)
  _update_emission_counts_mp   -- scatter-add (replaces sequential scan)

The transition-counts split drops the (num_splits - 1) cross-segment boundary
pairs, which is consistent with the boundary approximation already made in the
split forward/backward passes and negligible for large T.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from cscg import cscg_he_split_em


class CSCG(cscg_he_split_em.CSCG):
  """cscg_he_split_em with parallelised counts accumulation.

  Inherits from cscg_he_split_em so the EM forward/backward passes remain
  split.  Additionally replaces every remaining sequential scan in the M-step
  with either a segment-parallel scan or a direct scatter-add.
  """

  def __init__(
      self,
      n_clones: Sequence[int],
      pseudocount: float,
      n_actions: int = 4,
      seed: int = 0,
      batched: bool = True,
      use_bfloat16: bool = False,
      num_splits: int = 4,
  ):
    super().__init__(
        n_clones=n_clones,
        pseudocount=pseudocount,
        n_actions=n_actions,
        seed=seed,
        batched=batched,
        use_bfloat16=use_bfloat16,
        num_splits=num_splits,
    )

    # --- EM transition counts: split scan (same num_splits as fwd/bwd) -------
    self._update_transition_counts = jax.pmap(
        self._update_transition_counts_split,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(6,),
    )

    # --- EM emission counts: scatter-add (no scan) ---------------------------
    self._update_emission_counts = jax.pmap(
        self._update_emission_counts_scatter,
        axis_name="devices",
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(4,),
        devices=self._devices,
    )

    # --- Viterbi transition counts: scatter-add (no scan) --------------------
    self._update_transition_counts_mp = jax.pmap(
        self._update_transition_counts_mp_scatter,
        axis_name="devices",
        in_axes=(0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )

    # --- Viterbi emission counts: scatter-add (no scan) ----------------------
    self._update_emission_counts_mp = jax.pmap(
        self._update_emission_counts_mp_scatter,
        axis_name="devices",
        in_axes=(0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )

  @property
  def implementation(self) -> str:
    return f"he_split_em_2_{self._num_splits}"

  # ---------------------------------------------------------------------------
  # Split transition-counts accumulation
  # ---------------------------------------------------------------------------

  def _update_transition_counts_split(
      self,
      transition_matrices: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      mess_bwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
      obs_to_start_state_index: jnp.ndarray,
      max_clones: int,
  ):
    """Accumulate EM transition counts in parallel across num_splits segments.

    mess_fwd[n] and mess_bwd[n] are already in forward-time order (mess_bwd
    has been flipped in the training loop before this call).

    Each segment independently runs a short sequential scan of length
    seg_len - 1, then the per-segment counts are summed.  The num_splits - 1
    cross-boundary pairs are dropped (negligible for large T).
    """
    T = observations.shape[0]
    seg_len = T // self._num_splits

    T_pad = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )

    # Reshape leading time axis into [num_splits, seg_len, ...]
    fwd_s = mess_fwd.reshape(self._num_splits, seg_len, max_clones)
    bwd_s = mess_bwd.reshape(self._num_splits, seg_len, max_clones)
    obs_s = observations.reshape(self._num_splits, seg_len)
    act_s = actions.reshape(self._num_splits, seg_len)

    # Within each segment: pairs (n, n+1) for local n in [0, seg_len-2].
    # Cross-boundary pairs (local n = seg_len-1) are omitted.
    fwd_n  = fwd_s[:, :-1, :]   # [num_splits, seg_len-1, max_clones]
    bwd_n1 = bwd_s[:, 1:, :]    # [num_splits, seg_len-1, max_clones]
    obs_n  = obs_s[:, :-1]      # [num_splits, seg_len-1]
    obs_n1 = obs_s[:, 1:]       # [num_splits, seg_len-1]
    act_n  = act_s[:, :-1]      # [num_splits, seg_len-1]

    def one_segment(fwd_seg, bwd_seg, obs_seg_n, obs_seg_n1, act_seg):
      """Sequential scan over one segment, returning a local counts matrix."""
      inner_len = fwd_seg.shape[0]  # seg_len - 1
      counts_init = jnp.zeros(T_pad.shape, dtype=self._dtype)

      def step(counts, n):
        ts = jax.lax.dynamic_slice(
            T_pad[act_seg[n]],
            (obs_to_start_state_index[obs_seg_n[n]],
             obs_to_start_state_index[obs_seg_n1[n]]),
            (max_clones, max_clones),
        )
        cs = jax.lax.dynamic_slice(
            counts[act_seg[n]],
            (obs_to_start_state_index[obs_seg_n[n]],
             obs_to_start_state_index[obs_seg_n1[n]]),
            (max_clones, max_clones),
        )
        q = ts * jnp.outer(fwd_seg[n], bwd_seg[n])
        q /= q.sum()
        updated = jax.lax.dynamic_update_slice(
            counts,
            (cs + q)[None],
            (act_seg[n],
             obs_to_start_state_index[obs_seg_n[n]],
             obs_to_start_state_index[obs_seg_n1[n]]),
        )
        return updated, None

      local_counts, _ = jax.lax.scan(step, counts_init, jnp.arange(inner_len))
      return local_counts

    # Run all segments in parallel via vmap, then sum.
    all_counts = jax.vmap(one_segment, in_axes=(0, 0, 0, 0, 0))(
        fwd_n, bwd_n1, obs_n, obs_n1, act_n
    )  # [num_splits, n_actions, n_states_padded, n_states_padded]

    final_counts = all_counts.sum(axis=0)
    final_counts = jax.lax.psum(final_counts, axis_name="devices")
    final_counts = final_counts[:, :-max_clones, :-max_clones]
    return final_counts

  # ---------------------------------------------------------------------------
  # Scatter-add emission counts (EM)
  # ---------------------------------------------------------------------------

  def _update_emission_counts_scatter(
      self,
      emission_matrix: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      mess_bwd: jnp.ndarray,
      observations: jnp.ndarray,
      keep_clone_structure: bool = False,
  ):
    """Compute gamma and scatter-add into emission counts — no sequential scan."""
    gamma = mess_fwd * mess_bwd
    gamma /= gamma.sum(axis=1, keepdims=True)

    if keep_clone_structure:
      gamma = jnp.dot(gamma, self.n_clones_matrix)

    # emission_counts[s, obs[t]] += gamma[t, s]  for all t, s
    emission_counts = jnp.zeros(emission_matrix.shape, dtype=self._dtype)
    emission_counts = emission_counts.at[:, observations].add(gamma.T)
    emission_counts = jax.lax.psum(emission_counts, axis_name="devices")
    return emission_counts

  # ---------------------------------------------------------------------------
  # Scatter-add transition counts (Viterbi / max-product)
  # ---------------------------------------------------------------------------

  def _update_transition_counts_mp_scatter(
      self,
      transition_matrices: jnp.ndarray,
      actions: jnp.ndarray,
      states: jnp.ndarray,
  ):
    """Count hard (state_n, state_{n+1}) transitions — fully parallel."""
    counts = jnp.zeros(transition_matrices.shape, dtype=self._dtype)
    # actions[n-1], states[n-1] -> states[n] for n in 1..T-1
    counts = counts.at[actions[:-1], states[:-1], states[1:]].add(1.0)
    counts = jax.lax.psum(counts, axis_name="devices")
    return counts

  # ---------------------------------------------------------------------------
  # Scatter-add emission counts (Viterbi / max-product)
  # ---------------------------------------------------------------------------

  def _update_emission_counts_mp_scatter(
      self,
      emission_matrix: jnp.ndarray,
      states: jnp.ndarray,
      observations: jnp.ndarray,
  ):
    """Count hard (state, observation) pairs — fully parallel."""
    emission_counts = jnp.zeros(emission_matrix.shape, dtype=self._dtype)
    # Original scan started at n=1 (skipping step 0); preserve that behaviour.
    emission_counts = emission_counts.at[states[1:], observations[1:]].add(1.0)
    emission_counts = jax.lax.psum(emission_counts, axis_name="devices")
    return emission_counts
