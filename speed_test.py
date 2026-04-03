"""Speed test: original vs optimized for both cscg_he and cscg_se.

Optimizations applied
---------------------
Both HE and SE:
  __update_transition_counts_mp  (learn_viterbi_transition)
    scan of T at[a,i,j].add(1) steps  →  single vectorized scatter-add

SE only:
  __update_emission_counts       (learn_em_emission)
    scan of T outer-product steps  →  single matmul  gamma.T @ observations
    (clean direct replacement: soft observations are already float [T, E])
  __forward, __backward, __forward_mp, __update_transition_counts
    T sequential dot(emission_matrix, obs[n]) matvecs  →  one batched GEMM
    obs_liks = observations @ emission_matrix.T  computed once before scan
    __update_transition_counts also eliminates a second redundant dot per step

HE only:
  __update_emission_counts       (learn_em_emission)
    scan of column-adds  →  scatter  emission_counts.at[:,obs].add(gamma.T)
  __update_emission_counts_mp    (learn_viterbi_emission)
    scan of scalar updates  →  scatter  counts.at[states,obs].add(1)
  __forward_emission, __backward_emission, __forward_emission_mp
    T sequential emission_matrix[:, obs[n]] column gathers  →  one batch gather
    obs_liks = emission_matrix[:, observations].T  computed once before scan

Not optimized (both):
  __update_transition_counts     (learn_em_transition, fused backward+counts)
    genuine sequential Markov dependency; scatter does not apply.
    (SE version has obs_liks precompute to reduce per-step cost)

Usage:
    python speed_test.py
"""

import contextlib
import io
import time

import jax
import numpy as np

from cscg import cscg_he
from cscg import cscg_he_opt
from cscg import cscg_se
from cscg import cscg_se_opt

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
N_EMISSIONS    = 20
CLONES_PER_OBS = 10
N_ACTIONS      = 4
SEQ_LEN        = 80_000
BENCH_ITERS    = 6
SEED           = 0

N_CLONES = [CLONES_PER_OBS] * N_EMISSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_he(cls, seed=SEED):
    return cls(n_clones=N_CLONES, pseudocount=1e-4, n_actions=N_ACTIONS,
               batched=True, use_bfloat16=False, seed=seed)


def make_se(cls, seed=SEED):
    return cls(n_clones=N_CLONES, pseudocount=1e-4, n_actions=N_ACTIONS,
               batched=True, use_bfloat16=False, seed=seed)


def make_data_he(seq_len=SEQ_LEN, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, N_EMISSIONS, size=seq_len)
    act = rng.integers(0, N_ACTIONS,   size=seq_len)
    return obs, act


def make_data_se(seq_len=SEQ_LEN, seed=SEED):
    # SE uses one-hot / soft observations
    rng = np.random.default_rng(seed)
    obs_int = rng.integers(0, N_EMISSIONS, size=seq_len)
    obs = np.zeros((seq_len, N_EMISSIONS), dtype=np.float32)
    obs[np.arange(seq_len), obs_int] = 1.0
    act = rng.integers(0, N_ACTIONS, size=seq_len)
    return obs, act


def block():
    jax.effects_barrier()


def timed(fn, obs, act, n_iter):
    with contextlib.redirect_stderr(io.StringIO()):
        t0 = time.perf_counter()
        fn(obs, act, n_iter=n_iter)
        block()
        return time.perf_counter() - t0


def bench(label, fn_orig, fn_opt, obs, act, width=52):
    # warmup
    with contextlib.redirect_stderr(io.StringIO()):
        fn_orig(obs, act, n_iter=1); block()
        fn_opt (obs, act, n_iter=1); block()
    t_o = timed(fn_orig, obs, act, BENCH_ITERS) / BENCH_ITERS
    t_p = timed(fn_opt,  obs, act, BENCH_ITERS) / BENCH_ITERS
    tag = f"{t_o*1000:7.1f} → {t_p*1000:6.1f} ms   {t_o/t_p:.2f}x"
    print(f"    {label:{width}s}  {tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    devices = jax.devices()
    print(f"\nJAX backend  : {devices[0].platform.upper()}")
    print(f"JAX devices  : {devices}")
    print(f"N_EMISSIONS  : {N_EMISSIONS}  |  CLONES/OBS : {CLONES_PER_OBS}"
          f"  ({sum(N_CLONES)} states)  |  N_ACTIONS : {N_ACTIONS}")
    print(f"SEQ_LEN      : {SEQ_LEN:,}  |  BENCH_ITERS : {BENCH_ITERS}")

    he_obs, he_act = make_data_he()
    se_obs, se_act = make_data_se()

    # -----------------------------------------------------------------------
    # HE  (hard evidence)
    # -----------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  HE (hard evidence)  —  compiling warmup …", end=" ", flush=True)

    # Pre-build all four models so compilation is shared
    he_o = make_he(cscg_he.CSCG)
    he_p = make_he(cscg_he_opt.CSCG)

    print("done")
    print(f"  {'method':<52}  {'orig → opt ms':>22}  speedup")

    # learn_viterbi_transition  — opt: transition counts scatter
    bench("learn_viterbi_transition  [transition scatter]",
          he_o.learn_viterbi_transition,
          he_p.learn_viterbi_transition,
          he_obs, he_act)

    # learn_em_transition  — NOT optimized (control; fused backward+counts scan)
    bench("learn_em_transition       [NOT optimized, control]",
          he_o.learn_em_transition,
          he_p.learn_em_transition,
          he_obs, he_act)

    # learn_em_emission  — opt: emission scatter + obs_liks precompute
    bench("learn_em_emission         [scatter + obs_liks precompute]",
          he_o.learn_em_emission,
          he_p.learn_em_emission,
          he_obs, he_act)

    # learn_viterbi_emission  — opt: emission scatter + obs_liks precompute
    bench("learn_viterbi_emission    [scatter + obs_liks precompute]",
          he_o.learn_viterbi_emission,
          he_p.learn_viterbi_emission,
          he_obs, he_act)

    # -----------------------------------------------------------------------
    # SE  (soft evidence)
    # -----------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  SE (soft evidence)  —  compiling warmup …", end=" ", flush=True)

    se_o = make_se(cscg_se.CSCG)
    se_p = make_se(cscg_se_opt.CSCG)

    print("done")
    print(f"  {'method':<52}  {'orig → opt ms':>22}  speedup")

    # learn_viterbi_transition  — opt: transition scatter + obs_liks precompute
    bench("learn_viterbi_transition  [scatter + obs_liks precompute]",
          se_o.learn_viterbi_transition,
          se_p.learn_viterbi_transition,
          se_obs, se_act)

    # learn_em_transition  — opt: obs_liks precompute (scan carries remain sequential)
    bench("learn_em_transition       [obs_liks precompute in scan]",
          se_o.learn_em_transition,
          se_p.learn_em_transition,
          se_obs, se_act)

    # learn_em_emission  — opt: emission matmul + obs_liks precompute
    bench("learn_em_emission         [matmul + obs_liks precompute]",
          se_o.learn_em_emission,
          se_p.learn_em_emission,
          se_obs, se_act)

    print()


if __name__ == "__main__":
    main()
