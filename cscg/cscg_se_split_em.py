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

"""CSCG-SE with sequence-split EM forward/backward but non-split Viterbi.

The split approximation is applied only to the sum-product (EM) forward and
backward passes, which give the bulk of the per-iteration speedup.  The
max-product (Viterbi) forward pass and backtrace are left as the base class's
genuine sequential scans.

This avoids the segment-boundary discontinuity that corrupts Viterbi training
in the fully-split variant: without split Viterbi, the backtraced state
sequence is a coherent MAP path through the entire sequence, so
update_transition_counts_mp accumulates only real transitions.

Rebinds (split):
  _forward    -- EM sum-product forward (learn_em_transition)
  _backward   -- EM sum-product backward (learn_em_emission)

Unchanged from base cscg_se (non-split):
  _forward_mp                  -- Viterbi max-product forward
  _backtrace                   -- Viterbi backtrace
  _update_transition_counts    -- fused backward+counts scan (Markov dep.)
  _update_transition_counts_mp -- scatter-add (already parallel)
  _update_emission_counts      -- sequential scan
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from cscg import cscg_se


class CSCG(cscg_se.CSCG):
  """CSCG-SE with split EM passes and non-split Viterbi passes.

  Inherits from cscg_se (not cscg_se_opt).  Only _forward and _backward are
  replaced with split versions; the Viterbi pair (_forward_mp, _backtrace)
  remains the base class's exact sequential implementation.
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
    )
    self._num_splits = num_splits
    assert num_splits > 0, "num_splits must be positive"

    # Rebind only the EM sum-product passes.
    # _forward_mp and _backtrace are intentionally left as the base class's
    # non-split versions so that Viterbi training produces coherent paths.
    self._forward = jax.pmap(
        self._forward_split,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )
    self._backward = jax.pmap(
        self._backward_split,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )

  @property
  def implementation(self) -> str:
    return f"se_split_em_{self._num_splits}"

  # ---------------------------------------------------------------------------
  # Helper
  # ---------------------------------------------------------------------------

  def _split(self, *arrays):
    """Reshape each array's leading axis [T] -> [num_splits, seg_len, ...]."""
    return [
        a.reshape(self._num_splits, a.shape[0] // self._num_splits, *a.shape[1:])
        for a in arrays
    ]

  # ---------------------------------------------------------------------------
  # forward  (EM forward, sum-product)
  # ---------------------------------------------------------------------------

  def _forward_split(
      self,
      transition_matrices,
      emission_matrix,
      pi,
      observations,
      actions,
  ):
    obs_s, act_s = self._split(observations, actions)
    T_t = transition_matrices.transpose(0, 2, 1)
    num_states = emission_matrix.shape[0]

    def one_segment(obs_seg, act_seg):
      obs_liks = jnp.dot(obs_seg, emission_matrix.T)  # [seg_len, num_states]
      init = pi * obs_liks[0]
      p0 = init.sum()
      init = init / p0

      def step(msg, n):
        m = jnp.matmul(T_t[act_seg[n - 1]], msg) * obs_liks[n]
        p = m.sum()
        return m / p, (m / p, p)

      _, (msgs, ps) = jax.lax.scan(step, init, jnp.arange(1, obs_seg.shape[0]))
      return jnp.log2(jnp.hstack((p0, ps))), jnp.concatenate((init[None], msgs))

    log2_liks, messages = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return log2_liks.reshape(-1), messages.reshape(-1, num_states)

  # ---------------------------------------------------------------------------
  # backward  (EM backward, sum-product)
  # ---------------------------------------------------------------------------

  def _backward_split(
      self,
      transition_matrices,
      emission_matrix,
      observations,
      actions,
  ):
    obs_s, act_s = self._split(observations, actions)
    num_states = emission_matrix.shape[0]

    def one_segment(obs_seg, act_seg):
      obs_liks = jnp.dot(obs_seg, emission_matrix.T)  # [seg_len, num_states]
      sl = obs_seg.shape[0]
      init = jnp.ones(num_states, dtype=self._dtype)
      init = init / init.sum()

      def step(msg, n):
        m = jnp.matmul(
            transition_matrices[act_seg[n]], msg * obs_liks[n + 1]
        )
        return m / m.sum(), m / m.sum()

      _, msgs = jax.lax.scan(step, init, jnp.arange(sl - 2, -1, -1))
      return jnp.concatenate((init[None], msgs))

    msgs_split = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return jnp.flip(msgs_split, axis=0).reshape(-1, num_states)
