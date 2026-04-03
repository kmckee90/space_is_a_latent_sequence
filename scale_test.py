"""Scaling test: does speedup grow with sequence length?

For each optimized method we sweep T over several sequence lengths and record
the orig / opt ratio.  If speedups scale with T we expect the ratio to increase
(or at least hold) as T grows; if the forward/backward scan dominates at all T
the ratio will plateau.

Usage:
    python scale_test.py
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
# Configuration
# ---------------------------------------------------------------------------
N_EMISSIONS    = 20
CLONES_PER_OBS = 10
N_ACTIONS      = 4
BENCH_ITERS    = 3
SEED           = 0

N_CLONES = [CLONES_PER_OBS] * N_EMISSIONS

# Sequence lengths to sweep (total; split evenly across all GPUs)
SEQ_LENS = [10_000, 40_000, 160_000, 640_000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_he(cls, seed=SEED):
    return cls(n_clones=N_CLONES, pseudocount=1e-4, n_actions=N_ACTIONS,
               batched=True, use_bfloat16=False, seed=seed)


def make_se(cls, seed=SEED):
    return cls(n_clones=N_CLONES, pseudocount=1e-4, n_actions=N_ACTIONS,
               batched=True, use_bfloat16=False, seed=seed)


def make_data_he(seq_len, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, N_EMISSIONS, size=seq_len)
    act = rng.integers(0, N_ACTIONS,   size=seq_len)
    return obs, act


def make_data_se(seq_len, seed=SEED):
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


def bench_pair(fn_orig, fn_opt, obs, act):
    """Return (t_orig_ms, t_opt_ms, speedup) after warmup."""
    with contextlib.redirect_stderr(io.StringIO()):
        fn_orig(obs, act, n_iter=1); block()
        fn_opt (obs, act, n_iter=1); block()
    t_o = timed(fn_orig, obs, act, BENCH_ITERS) / BENCH_ITERS
    t_p = timed(fn_opt,  obs, act, BENCH_ITERS) / BENCH_ITERS
    return t_o * 1000, t_p * 1000, t_o / t_p


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

def sweep(label, methods, make_data_fn, make_orig, make_opt, seq_lens):
    """
    methods: list of (short_name, attr_name) pairs
    """
    n_devs = jax.device_count()
    print(f"\n{'═'*80}")
    print(f"  {label}")
    print(f"{'─'*80}")

    # Build models once (reused across lengths — recompile per new shape is OK)
    orig = make_orig()
    opt  = make_opt()

    # Header
    col_w = 12
    lens_hdr = "  ".join(f"{T//1000:>6}k" for T in seq_lens)
    print(f"  {'method':<38}  {lens_hdr}")
    print(f"  {'':38}  " +
          "  ".join(f"{'orig→opt':>6}" for _ in seq_lens))
    print(f"{'─'*80}")

    for short_name, attr in methods:
        fn_o = getattr(orig, attr)
        fn_p = getattr(opt,  attr)
        row_vals = []
        for T in seq_lens:
            obs, act = make_data_fn(T)
            t_o, t_p, sp = bench_pair(fn_o, fn_p, obs, act)
            row_vals.append(f"{sp:>5.2f}x")
        print(f"  {short_name:<38}  {'  '.join(row_vals)}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    devices = jax.devices()
    n_devs  = len(devices)
    print(f"\nJAX backend  : {devices[0].platform.upper()}")
    print(f"JAX devices  : {devices}")
    print(f"N_EMISSIONS  : {N_EMISSIONS}  |  CLONES/OBS : {CLONES_PER_OBS}"
          f"  ({sum(N_CLONES)} states)  |  N_ACTIONS : {N_ACTIONS}")
    print(f"SEQ_LENS     : {[f'{T:,}' for T in SEQ_LENS]}  (per device: "
          f"{[f'{T//n_devs:,}' for T in SEQ_LENS]})")
    print(f"BENCH_ITERS  : {BENCH_ITERS}")

    he_methods = [
        ("learn_viterbi_transition [scatter]",        "learn_viterbi_transition"),
        ("learn_em_transition      [control]",         "learn_em_transition"),
        ("learn_em_emission        [scatter+obs_liks]","learn_em_emission"),
        ("learn_viterbi_emission   [scatter+obs_liks]","learn_viterbi_emission"),
    ]

    se_methods = [
        ("learn_viterbi_transition [scatter+obs_liks]","learn_viterbi_transition"),
        ("learn_em_transition      [obs_liks in scan]","learn_em_transition"),
        ("learn_em_emission        [matmul+obs_liks]", "learn_em_emission"),
    ]

    print("\nBuilding HE models …", end=" ", flush=True)
    sweep(
        "HE (hard evidence)  —  speedup by sequence length",
        he_methods,
        make_data_he,
        lambda: make_he(cscg_he.CSCG),
        lambda: make_he(cscg_he_opt.CSCG),
        SEQ_LENS,
    )

    print("Building SE models …", end=" ", flush=True)
    sweep(
        "SE (soft evidence)  —  speedup by sequence length",
        se_methods,
        make_data_se,
        lambda: make_se(cscg_se.CSCG),
        lambda: make_se(cscg_se_opt.CSCG),
        SEQ_LENS,
    )


if __name__ == "__main__":
    main()
