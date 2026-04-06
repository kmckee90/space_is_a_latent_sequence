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

"""CSCG-HE with sequence-split EM forward/backward but non-split Viterbi.

The split approximation is applied only to the sum-product (EM) forward and
backward passes, which give the bulk of the per-iteration speedup.  The
max-product (Viterbi) forward pass and backtrace are left as the base class's
genuine sequential scans.

This avoids the segment-boundary discontinuity that corrupts Viterbi training
in the fully-split variant: without split Viterbi, the backtraced state
sequence is a coherent MAP path through the entire sequence, so
update_transition_counts_mp accumulates only real transitions.

Rebinds (split):
  _forward           -- EM sum-product forward (learn_em_transition)
  _backward          -- EM sum-product backward (learn_em_transition)
  _forward_emission  -- EM emission forward (learn_em_emission)
  _backward_emission -- EM emission backward (learn_em_emission)

Unchanged from base cscg_he (non-split):
  _forward_mp                              -- Viterbi max-product forward
  _backtrace                               -- Viterbi backtrace
  _forward_emission_mp                     -- Viterbi emission forward
  _backtrace_emission                      -- Viterbi emission backtrace
  _update_transition_counts                -- fused backward+counts (Markov dep.)
  _update_transition_counts_mp             -- scatter-add (already parallel)
  _update_transition_counts_given_emission -- fused backward+counts scan
  _update_emission_counts                  -- scatter-add (already parallel)
  _update_emission_counts_mp               -- scatter-add (already parallel)
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from cscg import cscg_he


class CSCG(cscg_he.CSCG):
  """CSCG-HE with split EM passes and non-split Viterbi passes.

  Inherits from cscg_he (not cscg_he_opt).  Only the sum-product forward and
  backward passes are replaced with split versions; the max-product (Viterbi)
  pair remains the base class's exact sequential implementation.
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
    # _forward_mp, _backtrace, _forward_emission_mp, _backtrace_emission are
    # intentionally left as the base class's non-split versions so that
    # Viterbi training produces coherent paths.
    self._forward = jax.pmap(
        self._forward_split,
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0, 0),
        static_broadcasted_argnums=(6,),
    )
    self._backward = jax.pmap(
        self._backward_split,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(5,),
    )
    self._forward_emission = jax.pmap(
        self._forward_emission_split,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )
    self._backward_emission = jax.pmap(
        self._backward_emission_split,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )

  @property
  def implementation(self) -> str:
    return f"he_split_em_{self._num_splits}"

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
      pi,
      observations,
      actions,
      obs_to_start_state_index,
      masked_multiplier,
      max_clones: int,
  ):
    obs_s, act_s = self._split(observations, actions)
    T_t = jnp.pad(
        transition_matrices.transpose(0, 2, 1),
        ((0, 0), (0, max_clones), (0, max_clones)),
    )

    def one_segment(obs_seg, act_seg):
      seg_len = obs_seg.shape[0]
      init = jax.lax.dynamic_slice(
          pi, (obs_to_start_state_index[obs_seg[0]],), (max_clones,)
      ) * masked_multiplier[obs_seg[0]]
      p0 = init.sum()
      init = init / p0

      def step(msg, n):
        ts = jax.lax.dynamic_slice(
            T_t[act_seg[n - 1]],
            (obs_to_start_state_index[obs_seg[n]],
             obs_to_start_state_index[obs_seg[n - 1]]),
            (max_clones, max_clones),
        )
        m = jnp.matmul(ts, msg) * masked_multiplier[obs_seg[n]]
        p = m.sum()
        return m / p, (m / p, p)

      _, (msgs, ps) = jax.lax.scan(step, init, jnp.arange(1, seg_len))
      return jnp.log2(jnp.hstack((p0, ps))), jnp.concatenate((init[None], msgs))

    log2_liks, messages = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return log2_liks.reshape(-1), messages.reshape(-1, max_clones)

  # ---------------------------------------------------------------------------
  # backward  (EM backward, sum-product)
  # ---------------------------------------------------------------------------

  def _backward_split(
      self,
      transition_matrices,
      observations,
      actions,
      obs_to_start_state_index,
      masked_multiplier,
      max_clones: int,
  ):
    obs_s, act_s = self._split(observations, actions)
    T_pad = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )

    def one_segment(obs_seg, act_seg):
      sl = obs_seg.shape[0]
      init = (jnp.ones(max_clones, dtype=self._dtype)
              * masked_multiplier[obs_seg[sl - 1]])
      init = init / init.sum()

      def step(msg, n):
        ts = jax.lax.dynamic_slice(
            T_pad[act_seg[n]],
            (obs_to_start_state_index[obs_seg[n]],
             obs_to_start_state_index[obs_seg[n + 1]]),
            (max_clones, max_clones),
        )
        m = jnp.matmul(ts, msg) * masked_multiplier[obs_seg[n]]
        return m / m.sum(), m / m.sum()

      _, msgs = jax.lax.scan(step, init, jnp.arange(sl - 2, -1, -1))
      return jnp.concatenate((init[None], msgs))

    msgs_split = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return jnp.flip(msgs_split, axis=0).reshape(-1, max_clones)

  # ---------------------------------------------------------------------------
  # forward_emission  (EM emission forward)
  # ---------------------------------------------------------------------------

  def _forward_emission_split(
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
      obs_liks = emission_matrix[:, obs_seg].T  # [seg_len, num_states]
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
  # backward_emission  (EM emission backward)
  # ---------------------------------------------------------------------------

  def _backward_emission_split(
      self,
      transition_matrices,
      emission_matrix,
      observations,
      actions,
  ):
    obs_s, act_s = self._split(observations, actions)
    num_states = emission_matrix.shape[0]

    def one_segment(obs_seg, act_seg):
      obs_liks = emission_matrix[:, obs_seg].T  # [seg_len, num_states]
      init = jnp.ones(num_states, dtype=self._dtype)
      init = init / init.sum()

      def step(msg, n):
        m = jnp.matmul(
            transition_matrices[act_seg[n]], msg * obs_liks[n + 1]
        )
        return m / m.sum(), m / m.sum()

      _, msgs = jax.lax.scan(step, init, jnp.arange(obs_seg.shape[0] - 2, -1, -1))
      return jnp.concatenate((init[None], msgs))

    msgs_split = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return jnp.flip(msgs_split, axis=0).reshape(-1, num_states)
