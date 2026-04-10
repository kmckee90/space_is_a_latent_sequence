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

"""Standalone CSCG-SE with parallelised split EM and scatter-add M-step.

Drop-in replacement for cscg_se.CSCG that collapses the four-class
inheritance chain (cscg → cscg_se → cscg_se_split_base → cscg_se_split_em_2)
into a single file.  Code and annotations are kept identical to cscg_se
everywhere possible so that a diff highlights only the differences.

Methods that differ from cscg_se
─────────────────────────────────
__init__
  • Accepts an additional ``num_splits`` argument (default 8).
  • ``self._forward``, ``self._backward``, ``self._forward_mp``,
    ``self._backtrace`` are pmap-wrapped around the split variants
    instead of the sequential originals.
  • ``self._update_transition_counts`` is pmap-wrapped around the split
    segment-parallel fused backward+counts scan.
  • ``self._update_emission_counts`` is pmap-wrapped around a matmul
    (gamma.T @ observations) instead of a sequential scan.
  • ``self._update_transition_counts_mp`` is pmap-wrapped around a
    scatter-add instead of a sequential scan.
  (The sequential originals — ``__forward``, ``__backward``, etc. — are still
  present; they are called directly by ``bps``, ``decode``, and friends.)

implementation (property)
  Returns ``f"se_split_{self._num_splits}"`` instead of ``"se"``.

_split                              NEW — helper to reshape T → [splits, seg, ...]
_forward_split                      NEW — vmapped segment-wise EM forward
_backward_split                     NEW — vmapped segment-wise EM backward
_update_transition_counts_split     NEW — split fused backward+counts
_update_emission_counts_scatter     NEW — matmul emission counts (no scan)
_update_transition_counts_mp_scatter
                                    NEW — scatter-add Viterbi transition counts
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from cscg import cscg
from cscg import cscg_se_utils as cscg_utils


class CSCG(cscg.CSCG):
  """A class implementing chmm with soft evidence using JAX.

  Attributes:
    num_states: The number of states in the graph.
    num_emissions: The number of unique emissions.
    emission_matrix: The emission matrix.
    counts_matrix: The counts matrix.
    transition_matrix: The transition matrix.
  """

  def __init__(
      self,
      n_clones: Sequence[int],
      pseudocount: float,
      n_actions: int = 4,
      seed: int = 0,
      batched: bool = True,
      use_bfloat16: bool = False,
      num_splits: int = 8,                                         # NEW
  ):
    """Init.

    Args:
      n_clones: A list containing number of clones for each emission.
      pseudocount: Smoothing factor. Should be non-negative.
      n_actions: Number of actions.
      seed: A seed for the random state.
      batched: Whether to use batched message computation.
      use_bfloat16: Whether to use bfloat16 for computation.
      num_splits: Number of segments to split the sequence into for parallel
        EM forward/backward and counts accumulation.  T must be divisible by
        num_splits.                                                 # NEW
    """
    self._key = jax.random.PRNGKey(seed)
    assert pseudocount >= 0.0, "The pseudocount should be non-negative"
    self._pseudocount = pseudocount
    self._n_clones = np.array(n_clones, dtype=int)
    self._num_actions = n_actions
    self._num_states = self._n_clones.sum()
    self._num_emissions = len(self._n_clones)
    self._n_clones_matrix = (
        None  # used only if keep clone structure is true for E learning
    )

    if batched:
      self._batched = True
      self._num_devices = jax.device_count()
      self._local_device_count = jax.local_device_count()
      self._local_devices = jax.local_devices()
      self._devices = jax.devices()
    else:
      self._batched = False
      self._num_devices = 1
      self._local_device_count = 1
      self._local_devices = [jax.local_devices()[0]]
      self._devices = [jax.local_devices()[0]]

    self._use_bfloat16 = use_bfloat16
    if self._use_bfloat16:
      self._dtype = jnp.bfloat16
    else:
      self._dtype = jnp.float32

    self._num_splits = num_splits                                   # NEW
    assert num_splits > 0, "num_splits must be positive"           # NEW

    # Brodcast necessary variables to local devices

    self._counts_matrix = cscg_utils.bcast_local_devices(
        jax.random.uniform(
            self._key,
            (self._num_actions, self._num_states, self._num_states),
            dtype=self._dtype,
        ),
        devices=self._local_devices,
    )
    self._emission_matrix = cscg_utils.bcast_local_devices(
        cscg_utils.get_default_emission_matrix(
            n_clones=self._n_clones, dtype=self._dtype
        ),
        devices=self._local_devices,
    )
    _, self._key = jax.random.split(self._key)
    self._pi_states = cscg_utils.bcast_local_devices(
        jnp.ones(self._num_states, dtype=self._dtype) / self._num_states,
        devices=self._local_devices,
    )

    self._pi_actions = cscg_utils.bcast_local_devices(
        jnp.ones(self._num_actions, dtype=self._dtype) / self._num_actions,
        devices=self._local_devices,
    )

    self._update_transition_matrix = jax.pmap(
        self.__update_transition_matrix,
        in_axes=(0,),
        out_axes=(0),
        static_broadcasted_argnums=(1,),
    )

    self._update_emission_matrix = jax.pmap(
        self.__update_emission_matrix,
        in_axes=(0,),
        out_axes=(0),
        static_broadcasted_argnums=(1, 2),
    )

    self._forward_mp = jax.pmap(
        self.__forward_mp,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )
    self._backtrace = jax.pmap(
        self.__backtrace,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )
    self._update_transition_counts_mp = jax.pmap(       # CHANGED: scatter-add
        self._update_transition_counts_mp_scatter,
        axis_name="devices",
        in_axes=(0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )
    self._forward = jax.pmap(                           # CHANGED: split vmap
        self._forward_split,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )
    self._backward = jax.pmap(                          # CHANGED: split vmap
        self._backward_split,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )
    self._update_transition_counts = jax.pmap(          # CHANGED: split fused
        self._update_transition_counts_split,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )
    self._update_emission_counts = jax.pmap(            # CHANGED: matmul
        self._update_emission_counts_scatter,
        axis_name="devices",
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=4,
        devices=self._devices,
    )

    self._transition_matrix = self._update_transition_matrix(
        self._counts_matrix, self._pseudocount
    )

  @property
  def implementation(self) -> str:
    return f"se_split_{self._num_splits}"               # CHANGED

  @property
  def batched(self) -> int:
    return self._batched

  @property
  def num_states(self) -> int:
    return self._num_states

  @property
  def num_emissions(self) -> int:
    return self._num_emissions

  @property
  def emission_matrix(self) -> np.ndarray:
    return jax.device_get(self._emission_matrix[0])

  @property
  def emission_matrix_default(self) -> np.ndarray:
    return cscg_utils.get_default_emission_matrix(
        n_clones=self._n_clones, dtype=self._dtype
    )

  @property
  def counts_matrix(self) -> np.ndarray:
    return jax.device_get(self._counts_matrix[0])

  @property
  def transition_matrix(self) -> np.ndarray:
    return jax.device_get(self._transition_matrix[0])

  @property
  def pseudocount(self) -> float:
    return self._pseudocount

  @property
  def n_clones(self) -> np.ndarray:
    return self._n_clones

  def set_counts_matrix(self, counts_matrix: np.ndarray) -> None:
    """Set the counts matrix."""

    assert counts_matrix.shape == (
        self._num_actions,
        self._num_states,
        self._num_states,
    ), (
        "Counts matrix dimensions are invalid, expected shape"
        f" {self._num_actions, self._num_states, self._num_states}"
    )
    self._counts_matrix = cscg_utils.bcast_local_devices(
        counts_matrix, devices=self._local_devices
    )
    self._transition_matrix = self._update_transition_matrix(
        self._counts_matrix, self._pseudocount
    )

  def set_pseudocount(self, pseudocount: float) -> None:
    assert pseudocount >= 0.0, "The pseudocount should be non-negative"
    self._pseudocount = pseudocount
    self._transition_matrix = self._update_transition_matrix(
        self._counts_matrix, self._pseudocount
    )

  def __update_transition_matrix(self, counts_matrix, pseudocount):
    """Update the transition matrix given the accumulated counts matrix."""
    transition_matrix = counts_matrix + pseudocount
    norm = transition_matrix.sum(2, keepdims=True)
    norm = jnp.where(norm == 0, 1, norm)
    transition_matrix /= norm
    return transition_matrix

  def __update_emission_matrix(
      self, emission_counts, pseudocount, pseudocount_extra
  ):
    """Update the emission matrix."""
    emission_matrix = emission_counts + pseudocount + pseudocount_extra
    norm = emission_matrix.sum(1, keepdims=True)
    norm = jnp.where(norm == 0, 1, norm)
    emission_matrix /= norm
    return emission_matrix

  def create_n_clones_matrix(self):
    """Create a block-diagonal matrix version of n_clones."""
    clone_idxs = np.hstack(([0], self._n_clones.cumsum()))
    n_clones_matrix = np.zeros((self._num_states, self._num_states), dtype=int)

    for idx in range(1, len(clone_idxs)):
      st, en = clone_idxs[idx - 1], clone_idxs[idx]
      n_clones_matrix[st:en, st:en] = np.ones(
          (self._n_clones[idx - 1], self._n_clones[idx - 1])
      )
    return n_clones_matrix

  def bridge(
      self, state1: int, state2: int, max_steps: int = 100
  ) -> tuple[np.ndarray, np.ndarray]:
    """Find the path between state1 and state2."""

    pi_states = np.zeros(self._num_states, dtype=self._dtype)
    pi_states[state1] = 1

    _, mess_fwd = self.__forward_mp_all(
        self._transition_matrix[0],
        pi_states,
        self._pi_actions[0],
        state2,
        max_steps,
    )

    s_a = self.__backtrace_all(
        self._transition_matrix[0], mess_fwd, self._pi_actions[0], state2
    )

    return s_a  # pytype: disable=bad-return-type  # jnp-type

  def sample(self, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample observations and actions from the CHMM."""

    random_state = np.random.RandomState()
    assert length > 0
    state_loc = np.hstack(([0], self._n_clones)).cumsum(0)
    sample_x = np.zeros(length, dtype=int)

    p_a = jax.device_get(self._pi_actions[0])
    sample_a = random_state.choice(
        len(p_a), size=length, p=p_a
    )
    # Sample
    transition_matrix = jax.device_get(self._transition_matrix[0])
    p_h = jax.device_get(self._pi_states[0])
    for t in range(length):
      h = random_state.choice(len(p_h), p=p_h)
      sample_x[t] = np.digitize(h, state_loc) - 1
      p_h = transition_matrix[sample_a[t], h]
    return np.array(sample_x), np.array(sample_a)

  def bps(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ):
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""
    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_emissions=self._num_emissions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        dtype=self._dtype,
        training=False,
    )

    if emission_matrix is None:
      emission_matrix = self._emission_matrix[0]
    emission_matrix = jnp.array(emission_matrix, dtype=self._dtype)

    log2_lik, _ = self.__forward(
        self._transition_matrix[0],
        emission_matrix,
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
    )
    return -log2_lik.flatten()

  def bps_viterbi(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ):
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""
    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_emissions=self._num_emissions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        dtype=self._dtype,
        training=False,
    )

    if emission_matrix is None:
      emission_matrix = self._emission_matrix[0]
    emission_matrix = jnp.array(emission_matrix, dtype=self._dtype)

    log2_lik, _ = self.__forward_mp(
        self._transition_matrix[0],
        emission_matrix,
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
    )
    return -log2_lik.flatten()

  def learn_em_transition(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
      n_iter: int = 100,
      term_early: bool = True,
  ):
    """Run EM training, keeping E deterministic and fixed, learning T."""

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_emissions=self._num_emissions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        dtype=self._dtype,
    )

    if emission_matrix is not None:
      self._emission_matrix = cscg_utils.bcast_local_devices(
          np.array(emission_matrix, dtype=self._dtype),
          devices=self._local_devices,
      )

    convergence = []
    pbar = tqdm.trange(n_iter, position=0)
    log2_lik_old = -np.inf

    for _ in pbar:
      # E step ----
      # forward messages
      log2_lik, mess_fwd = self._forward(
          self._transition_matrix,
          self._emission_matrix,
          self._pi_states,
          observations,
          actions,
      )

      # compute backward messages and update counts matrix
      self._counts_matrix = self._update_transition_counts(
          self._transition_matrix,
          self._emission_matrix,
          mess_fwd,
          observations,
          actions,
      )

      # M step ----
      self._transition_matrix = self._update_transition_matrix(
          self._counts_matrix, self._pseudocount
      )
      # Convergence check
      if self._use_bfloat16:
        log2_lik = log2_lik.astype(np.float32)
      convergence.append(-log2_lik.mean())
      pbar.set_postfix(train_bps=convergence[-1])
      if log2_lik.mean() <= log2_lik_old:
        if term_early:
          break
      log2_lik_old = log2_lik.mean()
    return convergence

  def learn_em_emission(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      n_iter: int = 100,
      keep_clone_structure: bool = False,
      emission_matrix_init: Optional[np.ndarray] = None,
      random_init: bool = False,
      noise_seed: int = 0,
      pseudocount_extra: float = 1e-20,
  ):
    """Run EM training, keeping E deterministic and fixed, learning T."""
    # Initialize n_clones matrix if keep_clone_structure is True
    if keep_clone_structure:
      self._n_clones_matrix = self.create_n_clones_matrix()

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_emissions=self._num_emissions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        dtype=self._dtype,
    )

    if emission_matrix_init is None:
      if random_init:
        emission_matrix = jax.random.uniform(
            jax.random.PRNGKey(noise_seed),
            shape=(self._num_states, self._num_emissions),
            dtype=self._dtype,
        )
      else:
        emission_matrix = jnp.ones(
            (self._num_states, self._num_emissions), dtype=self._dtype
        )
      emission_matrix /= emission_matrix.sum(axis=1, keepdims=True)
    else:
      emission_matrix = jnp.array(emission_matrix_init, dtype=self._dtype)

    self._emission_matrix = cscg_utils.bcast_local_devices(
        emission_matrix, devices=self._local_devices
    )

    convergence = []
    pbar = tqdm.trange(n_iter, position=0)
    log2_lik_old = -np.inf

    for _ in pbar:
      # E step ----
      # forward messages
      log2_lik, mess_fwd = self._forward(
          self._transition_matrix,
          self._emission_matrix,
          self._pi_states,
          observations,
          actions,
      )

      mess_bwd = self._backward(
          self._transition_matrix, self._emission_matrix, observations, actions
      )
      mess_bwd = jnp.flip(mess_bwd, axis=1)

      # Update emission counts
      emission_counts = self._update_emission_counts(
          self._emission_matrix,
          mess_fwd,
          mess_bwd,
          observations,
          keep_clone_structure,
      )

      # M step ----
      self._emission_matrix = self._update_emission_matrix(
          emission_counts, self._pseudocount, pseudocount_extra
      )

      # Convergence check
      if self._use_bfloat16:
        log2_lik = log2_lik.astype(np.float32)
      convergence.append(-log2_lik.mean())
      pbar.set_postfix(train_bps=convergence[-1])
      if log2_lik.mean() <= log2_lik_old:
        break
      log2_lik_old = log2_lik.mean()
    return convergence, jax.device_get(self._emission_matrix[0])

  def learn_viterbi_transition(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
      n_iter: int = 100,
  ):
    """Run Viterbi training, keeping E deterministic and fixed, learning T."""
    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_emissions=self._num_emissions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        dtype=self._dtype,
    )

    if emission_matrix is not None:
      self._emission_matrix = cscg_utils.bcast_local_devices(
          np.array(emission_matrix, dtype=self._dtype),
          devices=self._local_devices,
      )

    self.set_pseudocount(0.0)

    convergence = []
    pbar = tqdm.trange(n_iter, position=0)
    log2_lik_old = -jnp.inf

    for _ in pbar:
      # E step ----
      # forward messages
      log2_lik, mess_fwd = self._forward_mp(
          self._transition_matrix,
          self._emission_matrix,
          self._pi_states,
          observations,
          actions,
      )

      # backtrace
      states = self._backtrace(
          self._transition_matrix, mess_fwd, observations, actions
      )
      states = jnp.flip(states, axis=1)

      # update counts matrix
      self._counts_matrix = self._update_transition_counts_mp(
          self._transition_matrix, actions, states
      )

      # M step ----
      self._transition_matrix = self._update_transition_matrix(
          self._counts_matrix, self._pseudocount
      )
      # Convergence check
      if self._use_bfloat16:
        log2_lik = log2_lik.astype(np.float32)
      convergence.append(-log2_lik.mean())
      pbar.set_postfix(train_bps=convergence[-1])
      if log2_lik.mean() <= log2_lik_old:
        break
      log2_lik_old = log2_lik.mean()
    return convergence

  def decode(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ):
    """Compute the MAP assignment of latent variables using max-product message passing."""
    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_emissions=self._num_emissions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        dtype=self._dtype,
        training=False,
    )

    if emission_matrix is None:
      emission_matrix = self._emission_matrix[0]
    emission_matrix = jnp.array(emission_matrix, dtype=self._dtype)

    # forward messages
    log2_lik, mess_fwd = self.__forward_mp(
        self._transition_matrix[0],
        emission_matrix,
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
    )

    states = self.__backtrace(
        self._transition_matrix[0],
        mess_fwd,
        jnp.array(observations),
        jnp.array(actions),
    )
    states = jnp.flip(states, axis=0)

    return -log2_lik, states.flatten()

  # ---------------------------------------------------------------------------
  # NEW: sequence-split helpers
  # ---------------------------------------------------------------------------

  def _split(self, *arrays):
    """Reshape each array's leading axis [T] -> [num_splits, seg_len, ...]."""
    return [
        a.reshape(self._num_splits, a.shape[0] // self._num_splits, *a.shape[1:])
        for a in arrays
    ]

  # ---------------------------------------------------------------------------
  # NEW: split EM forward (sum-product)
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
  # NEW: split EM backward (sum-product)
  #
  # Returns messages in reverse-chronological order (same as __backward) so
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

    # msgs_split: [num_splits, seg_len, num_states], each in reverse scan order.
    # Reversing segment order gives global reverse-chronological layout.
    msgs_split = jax.vmap(one_segment, in_axes=(0, 0))(obs_s, act_s)
    return jnp.flip(msgs_split, axis=0).reshape(-1, num_states)

  # ---------------------------------------------------------------------------
  # NEW: split fused backward+counts (EM transition learning)
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
  # NEW: matmul emission counts (EM) — replaces sequential scan
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
      gamma = jnp.dot(gamma, self._n_clones_matrix)  # pytype: disable=wrong-arg-types  # jnp-type

    # emission_counts[s, e] = sum_t gamma[t, s] * observations[t, e]
    emission_counts = jnp.dot(gamma.T, observations)
    emission_counts = jax.lax.psum(emission_counts, axis_name="devices")
    return emission_counts

  # ---------------------------------------------------------------------------
  # NEW: scatter-add Viterbi transition counts — replaces sequential scan
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

  # ---------------------------------------------------------------------------
  # Below: internal methods identical to cscg_se — kept for bps/decode and
  # for easy diff comparison.  The training loop no longer calls these
  # directly; it calls the split/scatter variants above via self._forward etc.
  # ---------------------------------------------------------------------------

  def __forward(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      pi: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the forward messages."""

    transition_matrices = transition_matrices.transpose(0, 2, 1)
    sequence_len = observations.shape[0]
    initial_message = pi * jnp.dot(emission_matrix, observations[0])
    p_obs_0 = initial_message.sum()
    initial_message /= p_obs_0

    def one_step(message, n):
      new_message = jnp.dot(
          transition_matrices[actions[n - 1], :, :], message
      ) * jnp.dot(emission_matrix, observations[n])
      p_obs = new_message.sum()
      new_message /= p_obs
      return new_message, (new_message, p_obs)

    _, (messages, p_obs) = jax.lax.scan(
        one_step, initial_message, jnp.arange(1, sequence_len)
    )
    messages = jnp.concatenate((initial_message[None, :], messages))
    p_obs = jnp.hstack((p_obs_0, p_obs))
    log2_lik = jnp.log2(p_obs)

    return log2_lik, messages

  def __backward(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the backward messages."""

    sequence_len = observations.shape[0]

    initial_message = jnp.ones(emission_matrix.shape[0])
    initial_message /= initial_message.sum()

    def one_step(message, n):
      aij = actions[n - 1]

      # update message
      new_message = jnp.dot(
          transition_matrices[aij, :, :],
          message * jnp.dot(emission_matrix, observations[n]),
      )
      p_obs = new_message.sum()
      new_message /= p_obs

      return new_message, new_message

    _, messages = jax.lax.scan(
        one_step,
        initial_message,
        jnp.arange(sequence_len - 1, 0, -1),
    )
    messages = jnp.concatenate((initial_message[None, :], messages))
    return messages

  def __update_transition_counts(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """this function combines backward message passing with update counts."""
    sequence_len = observations.shape[0]
    self._counts_matrix = jnp.zeros(
        transition_matrices.shape, dtype=self._dtype
    )

    initial_message = jnp.ones(emission_matrix.shape[0], dtype=self._dtype)
    initial_message /= initial_message.sum()

    def one_step(inputs, n):
      counts_matrix, message = inputs

      aij = actions[n - 1]

      m_f = mess_fwd[n - 1]
      m_b = message * jnp.dot(
          emission_matrix, observations[n]
      )  # weighted backward messages

      q = m_f.reshape(-1, 1) * transition_matrices[aij] * m_b.reshape(1, -1)
      q /= q.sum()

      updated_counts_matrix = counts_matrix.at[aij].add(q)

      # update message
      new_message = jnp.dot(
          transition_matrices[aij, :, :],
          message * jnp.dot(emission_matrix, observations[n]),
      )
      p_obs = new_message.sum()
      new_message /= p_obs

      return (updated_counts_matrix, new_message), None

    (final_counts_matrix, _), _ = jax.lax.scan(
        one_step,
        (self._counts_matrix, initial_message),
        jnp.arange(sequence_len - 1, 0, -1),
    )

    final_counts_matrix = jax.lax.psum(final_counts_matrix, axis_name="devices")

    return final_counts_matrix

  def __update_emission_counts(
      self,
      emission_matrix: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      mess_bwd: jnp.ndarray,
      observations: jnp.ndarray,
      keep_clone_structure: bool = False,
  ):
    """Update emission counts."""
    sequence_len = observations.shape[0]

    gamma = mess_fwd * mess_bwd
    gamma /= gamma.sum(axis=1, keepdims=True)  # sum over latent states axis

    if keep_clone_structure:
      gamma = jnp.dot(gamma, self._n_clones_matrix)  # pytype: disable=wrong-arg-types  # jnp-type

    emission_counts_init = emission_matrix * 0

    def one_step(counts_matrix, n):
      counts_matrix += jnp.dot(gamma[n][:, None], observations[n][None, :])
      return counts_matrix, None

    emission_counts, _ = jax.lax.scan(
        one_step, emission_counts_init, jnp.arange(sequence_len)
    )

    emission_counts = jax.lax.psum(emission_counts, axis_name="devices")

    return emission_counts

  def __forward_mp(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      pi: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the forward messages."""

    transition_matrices = transition_matrices.transpose(0, 2, 1)
    sequence_len = observations.shape[0]

    initial_message = pi * jnp.dot(emission_matrix, observations[0])
    p_obs_0 = initial_message.max()
    initial_message /= p_obs_0

    def one_step(message, n):
      new_message = (
          transition_matrices[actions[n - 1], :, :] * message.reshape(1, -1)
      ).max(axis=1)
      new_message *= jnp.dot(emission_matrix, observations[n])
      p_obs = new_message.max()
      new_message /= p_obs
      return new_message, (new_message, p_obs)

    _, (messages, p_obs) = jax.lax.scan(
        one_step, initial_message, jnp.arange(1, sequence_len)
    )
    messages = jnp.concatenate((initial_message[None, :], messages))
    p_obs = jnp.hstack((p_obs_0, p_obs))
    log2_lik = jnp.log2(p_obs)

    return log2_lik, messages

  def __backtrace(
      self,
      transition_matrices: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the backtrace."""
    sequence_len = observations.shape[0]

    initial_state = mess_fwd[sequence_len - 1].argmax()

    def one_step(state, n):
      belief = mess_fwd[n] * transition_matrices[actions[n], :, state]
      new_state = belief.argmax()
      return new_state, new_state

    _, states = jax.lax.scan(
        one_step, initial_state, jnp.arange(sequence_len - 2, -1, -1)
    )
    states = jnp.hstack((initial_state, states))

    return states

  def __update_transition_counts_mp(
      self,
      transition_matrices: jnp.ndarray,
      actions: jnp.ndarray,
      states: jnp.ndarray,
  ):
    """This function updates counts matrix."""
    sequence_len = actions.shape[0]
    self._counts_matrix = jnp.zeros(
        transition_matrices.shape, dtype=self._dtype
    )

    def one_step(counts_matrix, n):
      aij = actions[n - 1]
      i, j = states[n - 1], states[n]
      updated_counts_matrix = counts_matrix.at[aij, i, j].add(1.0)

      return updated_counts_matrix, None

    final_counts_matrix, _ = jax.lax.scan(
        one_step, self._counts_matrix, jnp.arange(1, sequence_len)
    )

    final_counts_matrix = jax.lax.psum(final_counts_matrix, axis_name="devices")

    return final_counts_matrix

  def __forward_mp_all(
      self,
      transition_matrices: jnp.ndarray,
      pi: jnp.ndarray,
      pi_actions: jnp.ndarray,
      target_state: int,
      max_steps: int,
  ):
    """Compute the forward messages."""
    transition_matrices = transition_matrices.transpose(0, 2, 1)

    message = pi
    p_obs_0 = message.max()
    message /= p_obs_0

    log2_lik = [np.log2(p_obs_0)]
    mess_fwd = [message]

    transition_matrices_maxa = (
        transition_matrices * pi_actions.reshape(-1, 1, 1)
    ).max(axis=0)

    for _ in range(1, max_steps):
      message = (transition_matrices_maxa * message.reshape(1, -1)).max(axis=1)
      p_obs = message.max()
      assert p_obs > 0
      message /= p_obs
      log2_lik.append(np.log2(p_obs))
      mess_fwd.append(message)
      if message[target_state] > 0:
        break
    else:
      assert False, "Unable to find a bridging path"

    return jnp.array(log2_lik), jnp.array(mess_fwd, dtype=self._dtype)

  def __backtrace_all(
      self,
      transition_matrices: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      pi_actions: jnp.ndarray,
      target_state: int,
  ):
    """Compute the backtrace."""

    states = jnp.zeros(mess_fwd.shape[0], dtype=int)
    actions = jnp.zeros(mess_fwd.shape[0], dtype=int)

    # backward pass
    t = mess_fwd.shape[0] - 1
    # last action is irrelevant, use an invalid value
    actions = actions.at[t].set(-1)
    states = states.at[t].set(target_state)

    for t in range(mess_fwd.shape[0] - 2, -1, -1):
      belief = (
          mess_fwd[t].reshape(1, -1)
          * transition_matrices[:, :, states[t + 1]]
          * pi_actions.reshape(-1, 1)
      )
      a_s = jnp.argmax(belief.flatten())
      actions = actions.at[t].set(a_s // self._num_states)
      states = states.at[t].set(a_s % self._num_states)
    return actions, states
