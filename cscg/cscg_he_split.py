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

"""CSCG-HE with sequence-split parallel forward/backward passes.

Each device's T//num_devices-step shard is split into num_splits independent
segments. jax.vmap runs all segments' forward/backward scans in parallel on the
device, reducing the sequential scan length from T//num_devices to
T//(num_devices * num_splits).

This introduces a boundary approximation: each segment starts from the prior
pi rather than from the true forward message at the previous segment boundary.
For grid-world models this is benign because the forward message converges
quickly from pi once local observations disambiguate the location.

Unmodified (not split):
  _update_transition_counts     -- genuine sequential Markov dependency
  _update_transition_counts_mp  -- scatter-add, already parallel
  _update_emission_counts       -- scatter-add, already parallel
  _update_emission_counts_mp    -- scatter-add, already parallel
  _update_transition_counts_given_emission -- fused backward+counts scan
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from cscg import cscg_he_opt


class CSCG(cscg_he_opt.CSCG):
  """CSCG-HE opt with forward/backward scans split into parallel segments."""

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
    assert (
        num_splits > 0
    ), "num_splits must be positive"

    # Rebind the pmap'd functions that wrap sequential scans.
    # The scatter-based and fused-scan functions from the parent are kept.
    self._forward_mp = jax.pmap(
        self._forward_mp_split,
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0, 0),
        static_broadcasted_argnums=(6,),
    )
    self._backtrace = jax.pmap(
        self._backtrace_split,
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(6,),
    )
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
    self._forward_emission_mp = jax.pmap(
        self._forward_emission_mp_split,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )
    self._backtrace_emission = jax.pmap(
        self._backtrace_emission_split,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )

  @property
  def implementation(self) -> str:
    return f"he_split_{self._num_splits}"

  # ---------------------------------------------------------------------------
  # Helper
  # ---------------------------------------------------------------------------

  def _split(self, *arrays):
    """Reshape each array's leading axis [T] -> [num_splits, seg_len, ...]."""
    return [a.reshape(self._num_splits, a.shape[0] // self._num_splits, *a.shape[1:])
            for a in arrays]

  # ---------------------------------------------------------------------------
  # forward_mp  (Viterbi forward, max-product)
  # ---------------------------------------------------------------------------

  def _forward_mp_split(
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
      p0 = init.max()
      init = init / p0

      def step(msg, n):
        ts = jax.lax.dynamic_slice(
            T_t[act_seg[n - 1]],
            (obs_to_start_state_index[obs_seg[n]],
             obs_to_start_state_index[obs_seg[n - 1]]),
            (max_clones, max_clones),
        )
        m = (ts * msg).max(axis=1) * masked_multiplier[obs_seg[n]]
        p = m.max()
        return m / p, (m / p, p)

      _, (msgs, ps) = jax.lax.scan(step, init, jnp.arange(1, seg_len))
      return jnp.log2(jnp.hstack((p0, ps))), jnp.concatenate((init[None], msgs))

    log2_liks, messages = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    # log2_liks: [S, seg_len]  messages: [S, seg_len, max_clones]
    return log2_liks.reshape(-1), messages.reshape(-1, max_clones)

  # ---------------------------------------------------------------------------
  # backtrace  (Viterbi backtrace)
  # ---------------------------------------------------------------------------

  def _backtrace_split(
      self,
      transition_matrices,
      mess_fwd,
      observations,
      actions,
      obs_to_start_state_index,
      masked_multiplier,
      max_clones: int,
  ):
    seg_len = observations.shape[0] // self._num_splits
    obs_s, act_s = self._split(observations, actions)
    fwd_s = mess_fwd.reshape(self._num_splits, seg_len, max_clones)
    T_pad = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )

    def one_segment(fwd_seg, obs_seg, act_seg):
      sl = obs_seg.shape[0]
      init = (fwd_seg[sl - 1].argmax()
               + obs_to_start_state_index[obs_seg[sl - 1]])

      def step(state, n):
        ts = jax.lax.dynamic_slice(
            T_pad[act_seg[n], :, state],
            (obs_to_start_state_index[obs_seg[n]],),
            (max_clones,),
        )
        belief = fwd_seg[n] * ts * masked_multiplier[obs_seg[n]]
        new_state = belief.argmax() + obs_to_start_state_index[obs_seg[n]]
        return new_state, new_state

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
  #
  # Returns messages in reverse-chronological order (same as base class) so
  # the caller's jnp.flip(mess_bwd, axis=1) continues to work unchanged.
  # Each segment scans backward within itself; reversing segment order
  # reconstructs the global reverse-chronological layout.
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
      return jnp.concatenate((init[None], msgs))  # [seg_len, max_clones], reverse order

    # msgs_split: [num_splits, seg_len, max_clones], each in reverse scan order
    # Reversing segment order gives global reverse-chronological layout.
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
  # Same reverse-segment trick as _backward_split.
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

  # ---------------------------------------------------------------------------
  # forward_emission_mp  (Viterbi emission forward, max-product)
  # ---------------------------------------------------------------------------

  def _forward_emission_mp_split(
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
        m = (T_t[act_seg[n - 1]] * msg).max(axis=1) * obs_liks[n]
        p = m.max()
        return m / p, (m / p, p)

      _, (msgs, ps) = jax.lax.scan(step, init, jnp.arange(1, obs_seg.shape[0]))
      return jnp.log2(jnp.hstack((p0, ps))), jnp.concatenate((init[None], msgs))

    log2_liks, messages = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return log2_liks.reshape(-1), messages.reshape(-1, num_states)

  # ---------------------------------------------------------------------------
  # backtrace_emission  (Viterbi emission backtrace)
  # ---------------------------------------------------------------------------

  def _backtrace_emission_split(
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
