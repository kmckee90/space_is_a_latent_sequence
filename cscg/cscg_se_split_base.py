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

"""CSCG-SE with sequence-split parallel forward/backward passes.

Inherits directly from cscg_se (NOT cscg_se_opt) so that the only algorithmic
difference from the base model is the boundary approximation introduced by
splitting. This enables a clean ablation of the split effect without
confounding with the JAX optimizations in cscg_se_opt.

The split approximation: each segment starts from the prior pi rather than
from the true forward message at the previous segment boundary.

Unmodified from base cscg_se:
  _update_transition_counts     -- genuine sequential Markov dependency
  _update_transition_counts_mp  -- scatter-add, already parallel
  _update_emission_counts       -- sequential scan
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from cscg import cscg_se


class CSCG(cscg_se.CSCG):
  """CSCG-SE with forward/backward scans split into parallel segments.

  Inherits from cscg_se (not cscg_se_opt) so the only algorithmic difference
  from cscg_se is the boundary approximation introduced by splitting.
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

    # Rebind only the forward/backward scans with split versions.
    # All other methods (update_transition_counts, update_emission_counts, etc.)
    # remain as in the base cscg_se model — no opt contamination.
    self._forward_mp = jax.pmap(
        self._forward_mp_split,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )
    self._backtrace = jax.pmap(
        self._backtrace_split,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )
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
    return f"se_split_base_{self._num_splits}"

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
  # forward_mp  (Viterbi forward, max-product)
  # ---------------------------------------------------------------------------

  def _forward_mp_split(
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
      p0 = init.max()
      init = init / p0

      def step(msg, n):
        m = (T_t[act_seg[n - 1]] * msg).max(axis=1) * obs_liks[n]
        p = m.max()
        return m / p, (m / p, p)

      _, (msgs, ps) = jax.lax.scan(step, init, jnp.arange(1, obs_seg.shape[0]))
      return jnp.log2(jnp.hstack((p0, ps))), jnp.concatenate((init[None], msgs))

    log2_liks, messages = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return log2_liks.reshape(-1), messages.reshape(-1, num_states)

  # ---------------------------------------------------------------------------
  # backtrace  (Viterbi backtrace)
  # ---------------------------------------------------------------------------

  def _backtrace_split(
      self,
      transition_matrices,
      mess_fwd,
      observations,
      actions,
  ):
    num_states = transition_matrices.shape[1]
    seg_len = observations.shape[0] // self._num_splits
    obs_s, act_s = self._split(observations, actions)
    fwd_s = mess_fwd.reshape(self._num_splits, seg_len, num_states)

    def one_segment(fwd_seg, obs_seg, act_seg):
      sl = obs_seg.shape[0]
      init = fwd_seg[sl - 1].argmax()

      def step(state, n):
        belief = fwd_seg[n] * transition_matrices[act_seg[n], :, state]
        return belief.argmax(), belief.argmax()

      _, states = jax.lax.scan(step, init, jnp.arange(sl - 2, -1, -1))
      return jnp.hstack((init, states))

    states = jax.vmap(one_segment, in_axes=(0, 0, 0))(fwd_s, obs_s, act_s)
    return states.reshape(-1)

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
  #
  # Returns messages in reverse-chronological order (same as base class) so
  # the caller's jnp.flip(mess_bwd, axis=1) continues to work unchanged.
  # Each segment scans backward within itself; reversing segment order
  # reconstructs the global reverse-chronological layout.
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
      return jnp.concatenate((init[None], msgs))  # [seg_len, num_states], reverse order

    # msgs_split: [num_splits, seg_len, num_states], each in reverse scan order
    # Reversing segment order gives global reverse-chronological layout.
    msgs_split = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return jnp.flip(msgs_split, axis=0).reshape(-1, num_states)
