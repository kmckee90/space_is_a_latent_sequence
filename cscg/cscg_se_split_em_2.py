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

"""CSCG-SE split EM v2: split fused backward+counts + scatter-add elsewhere.

Extends cscg_se_split_em (which already splits the EM sum-product forward and
backward passes) by parallelising the remaining sequential operations.

New rebinds (relative to cscg_se_split_em / cscg_se):

  _update_transition_counts    -- split fused backward+counts into num_splits
                                   parallel segment scans (same granularity as
                                   the EM forward split)
  _update_emission_counts      -- replaced by gamma.T @ observations (no scan)
  _update_transition_counts_mp -- scatter-add (replaces sequential scan)

Key difference from the HE variant: SE's _update_transition_counts is a
*fused* backward+counts scan — there is no pre-computed separate backward
array.  The split version gives each segment a fresh uniform backward-message
initialisation, consistent with the boundary approximation already made in the
split forward pass.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from cscg import cscg_se_split_em


class CSCG(cscg_se_split_em.CSCG):
  """cscg_se_split_em with parallelised counts accumulation.

  Inherits from cscg_se_split_em so the EM forward/backward passes remain
  split.  Additionally replaces every remaining sequential scan in the M-step
  with either a segment-parallel scan or a direct scatter-add / matmul.
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

    # --- EM transition counts: split fused backward+counts -------------------
    self._update_transition_counts = jax.pmap(
        self._update_transition_counts_split,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )

    # --- EM emission counts: matmul (no scan) --------------------------------
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

  @property
  def implementation(self) -> str:
    return f"se_split_em_2_{self._num_splits}"

  # ---------------------------------------------------------------------------
  # Split fused backward+counts (EM transition learning)
  # ---------------------------------------------------------------------------

  def _update_transition_counts_split(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Fused backward+counts accumulation split into num_splits parallel segments.

    Each segment runs an independent backward scan of length seg_len - 1,
    starting from a uniform backward-message initialisation (the same boundary
    approximation used by the split forward pass).  Local counts matrices are
    summed at the end.  The num_splits - 1 cross-segment boundary pairs are
    dropped, consistent with the existing approximation and negligible for
    large T.

    Arguments mirror __update_transition_counts in cscg_se:
      transition_matrices -- [n_actions, n_states, n_states]
      emission_matrix     -- [n_states, n_emissions]
      mess_fwd            -- [T, n_states]   forward messages
      observations        -- [T, n_emissions] one-hot observations
      actions             -- [T]
    """
    T = mess_fwd.shape[0]
    num_states = emission_matrix.shape[0]
    seg_len = T // self._num_splits
    inner_len = seg_len - 1  # within-segment transition pairs

    # Reshape to [num_splits, seg_len, ...]
    fwd_s = mess_fwd.reshape(self._num_splits, seg_len, num_states)
    obs_s = observations.reshape(self._num_splits, seg_len, -1)
    act_s = actions.reshape(self._num_splits, seg_len)

    # At the original step n (T-1 down to 1):
    #   uses mess_fwd[n-1],  observations[n],  actions[n-1]
    # For local segment index i (0 .. seg_len-2):
    #   fwd_segs[s, i] = mess_fwd[s*seg_len + i]           → mess_fwd[n-1]
    #   obs_segs[s, i] = observations[s*seg_len + 1 + i]   → observations[n]
    #   act_segs[s, i] = actions[s*seg_len + i]             → actions[n-1]
    # Scan goes backward: i from seg_len-2 down to 0.
    fwd_segs = fwd_s[:, :-1, :]   # [num_splits, seg_len-1, num_states]
    obs_segs = obs_s[:, 1:, :]    # [num_splits, seg_len-1, num_emissions]
    act_segs = act_s[:, :-1]      # [num_splits, seg_len-1]

    def one_segment(fwd_seg, obs_seg, act_seg):
      """Fused backward+counts for one segment."""
      counts_init = jnp.zeros(transition_matrices.shape, dtype=self._dtype)
      init_msg = jnp.ones(num_states, dtype=self._dtype) / num_states

      def step(carry, i):
        counts, message = carry
        aij = act_seg[i]
        m_f = fwd_seg[i]
        obs_liks = jnp.dot(emission_matrix, obs_seg[i])
        m_b = message * obs_liks
        q = m_f.reshape(-1, 1) * transition_matrices[aij] * m_b.reshape(1, -1)
        q /= q.sum()
        updated_counts = counts.at[aij].add(q)
        new_message = jnp.dot(transition_matrices[aij], message * obs_liks)
        new_message /= new_message.sum()
        return (updated_counts, new_message), None

      (local_counts, _), _ = jax.lax.scan(
          step,
          (counts_init, init_msg),
          jnp.arange(inner_len - 1, -1, -1),  # backward through the segment
      )
      return local_counts

    # Run all segments in parallel via vmap, then sum.
    all_counts = jax.vmap(one_segment, in_axes=(0, 0, 0))(
        fwd_segs, obs_segs, act_segs
    )  # [num_splits, n_actions, n_states, n_states]

    final_counts = all_counts.sum(axis=0)
    final_counts = jax.lax.psum(final_counts, axis_name="devices")
    return final_counts

  # ---------------------------------------------------------------------------
  # Emission counts (EM): matmul replaces sequential scan
  # ---------------------------------------------------------------------------

  def _update_emission_counts_scatter(
      self,
      emission_matrix: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      mess_bwd: jnp.ndarray,
      observations: jnp.ndarray,
      keep_clone_structure: bool = False,
  ):
    """Compute gamma and accumulate emission counts via matmul — no scan.

    The original scan computes:
      emission_counts += gamma[t][:, None] @ observations[t][None, :]
    for each t, which is equivalent to gamma.T @ observations.
    """
    gamma = mess_fwd * mess_bwd
    gamma /= gamma.sum(axis=1, keepdims=True)

    if keep_clone_structure:
      gamma = jnp.dot(gamma, self._n_clones_matrix)  # pytype: disable=wrong-arg-types

    # emission_counts[s, e] = sum_t gamma[t, s] * observations[t, e]
    emission_counts = jnp.dot(gamma.T, observations)
    emission_counts = jax.lax.psum(emission_counts, axis_name="devices")
    return emission_counts

  # ---------------------------------------------------------------------------
  # Transition counts (Viterbi): scatter-add replaces sequential scan
  # ---------------------------------------------------------------------------

  def _update_transition_counts_mp_scatter(
      self,
      transition_matrices: jnp.ndarray,
      actions: jnp.ndarray,
      states: jnp.ndarray,
  ):
    """Count hard (state_n, state_{n+1}) transitions — fully parallel."""
    counts = jnp.zeros(transition_matrices.shape, dtype=self._dtype)
    counts = counts.at[actions[:-1], states[:-1], states[1:]].add(1.0)
    counts = jax.lax.psum(counts, axis_name="devices")
    return counts
