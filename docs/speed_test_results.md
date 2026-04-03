# Speed Test Results — Original vs Optimized

**Config:** 20 emissions · 10 clones/obs (200 states) · 4 actions · SEQ_LEN=10,000 · 6 bench iters

---

## HE (Hard Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | scatter in `__update_transition_counts_mp` | 4.3 | 2.2 | **1.94×** | 39.3 | 28.4 | **1.39×** |
| `learn_em_transition` | *not optimized (control)* | 11.2 | 10.9 | 1.03× | 55.2 | 55.1 | 1.00× |
| `learn_em_emission` | scatter + obs_liks precompute | 63.7 | 60.4 | 1.05× | 64.7 | 52.9 | **1.22×** |
| `learn_viterbi_emission` | scatter + obs_liks precompute | 34.6 | 33.0 | 1.05× | 37.8 | 32.4 | **1.17×** |

## SE (Soft Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | scatter + obs_liks precompute | 215.8 | 211.1 | 1.02× | 40.0 | 32.2 | **1.24×** |
| `learn_em_transition` | obs_liks precompute in scan | 772.8 | 768.5 | 1.01× | 40.2 | 39.1 | 1.03× |
| `learn_em_emission` | matmul + obs_liks precompute | 89.9 | 61.1 | **1.47×** | 63.9 | 52.6 | **1.21×** |

---

## Large config: 60 emissions · 30 clones/obs (1800 states) · 4 actions · SEQ_LEN=40,000 · 6 bench iters

### HE (Hard Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | scatter in `__update_transition_counts_mp` | 85.9 | 84.8 | 1.01× | 101.0 | 74.8 | **1.35×** |
| `learn_em_transition` | *not optimized (control)* | 89.5 | 86.5 | 1.04× | 73.2 | 73.1 | 1.00× |
| `learn_em_emission` | scatter + obs_liks precompute | 15118.1 | 13642.9 | **1.11×** | 432.0 | 379.5 | **1.14×** |
| `learn_viterbi_emission` | scatter + obs_liks precompute | 9310.3 | 8193.7 | **1.14×** | 201.4 | 169.8 | **1.19×** |

### SE (Soft Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | scatter + obs_liks precompute | 11243.5 | 11837.2 | 0.95× | 113.8 | 83.2 | **1.37×** |
| `learn_em_transition` | obs_liks precompute in scan | 55068.3 | 55970.7 | 0.98× | 94.4 | 90.4 | 1.04× |
| `learn_em_emission` | matmul + obs_liks precompute | 13787.1 | 10369.4 | **1.33×** | 427.0 | 384.3 | **1.11×** |

---

## Wide config: 120 emissions · 30 clones/obs (3600 states) · 4 actions · SEQ_LEN=10,000 · 6 bench iters

*Same T as small config, double the state count of large config — isolates state-space scaling.*

### HE (Hard Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | scatter in `__update_transition_counts_mp` | 106.3 | 110.5 | 0.96× | 24.2 | 14.5 | **1.67×** |
| `learn_em_transition` | *not optimized (control)* | 95.8 | 94.1 | 1.02× | 21.1 | 21.0 | 1.00× |
| `learn_em_emission` | scatter + obs_liks precompute | 15000.7 | 13367.5 | **1.12×** | 289.0 | 272.1 | 1.06× |
| `learn_viterbi_emission` | scatter + obs_liks precompute | 5628.8 | 5371.6 | 1.05× | 79.4 | 72.2 | **1.10×** |

### SE (Soft Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | scatter + obs_liks precompute | 7536.3 | 7483.0 | 1.01× | 26.6 | 21.2 | **1.25×** |
| `learn_em_transition` | obs_liks precompute in scan | 63278.9 | 77541.0 | 0.82× | 80.6 | 82.0 | 0.98× |
| `learn_em_emission` | matmul + obs_liks precompute | 28739.7 | 25187.2 | **1.14×** | 277.1 | 273.6 | 1.01× |

---

## HE split (num_splits=2): opt vs split — SEQ_LEN=10,000, 20 emissions · 10 clones (200 states)

*Compares `cscg_he_opt` (baseline) against `cscg_he_split` with num_splits=2.
Each device shard of T//8=1250 steps is split into 2 segments of 625 steps, run in parallel via `jax.vmap`.*

| Method | CPU opt (ms) | CPU split (ms) | CPU speedup | GPU opt (ms) | GPU split (ms) | GPU speedup |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | 2.3 | 63.0 | 0.02× | 33.6 | 33.4 | 1.00× |
| `learn_em_transition` | 12.4 | 67.2 | 0.22× | 68.2 | 46.8 | **1.30×** |
| `learn_em_emission` | 61.0 | 302.4 | 0.21× | 51.9 | 34.4 | **1.58×** |
| `learn_viterbi_emission` | 33.1 | 169.7 | 0.17× | 38.0 | 37.7 | 1.01× |

**CPU:** vmap over a batched scan adds enormous overhead on CPU — XLA cannot truly parallelize the scan iterations without wide SIMD support, and the batched scan compilation is much slower than a plain sequential scan.

**GPU:** EM methods (transition + emission) benefit from halving the scan depth. Viterbi methods gain little — `learn_viterbi_transition`'s bottleneck is the scatter-add (already fast), and `learn_viterbi_emission`'s backtrace has a data-dependent sequential structure that resists batching.

---

## HE split (num_splits=40): original vs split — GPU only, SEQ_LEN=40,000, 20 emissions · 10 clones (200 states)

*125 steps/segment per device (40,000 ÷ 8 devices ÷ 40 splits). Baseline is `cscg_he` original (no optimizations).*

| Method | Original (ms) | Split-40 (ms) | Speedup |
|--------|:---:|:---:|:---:|
| `learn_viterbi_transition` | 245.6 | 15.0 | **16.4×** |
| `learn_em_transition` | 243.3 | 75.2 | **3.2×** |
| `learn_em_emission` | 253.2 | 41.2 | **6.2×** |
| `learn_viterbi_emission` | 123.5 | 3.9 | **31.8×** |

At long T the scan depth is the dominant cost. Compare with the 1M-step results below. Splitting into 40 segments reduces each device's scan from 5000 to 125 steps — the speedups scale roughly with that reduction. `learn_em_transition` gains less because its count-update scan (fused backward+counts, genuine Markov dependency) is not split and becomes the new bottleneck.

---

## HE split (num_splits=40): original vs split — GPU only, SEQ_LEN=1,000,000, 20 emissions · 10 clones (200 states)

*3,125 steps/segment per device (1,000,000 ÷ 8 devices ÷ 40 splits). Baseline is `cscg_he` original.*

| Method | Original (ms) | Split-40 (ms) | Speedup |
|--------|:---:|:---:|:---:|
| `learn_viterbi_transition` | 5087.5 | 111.6 | **45.6×** |
| `learn_em_transition` | 5051.7 | 1393.1 | **3.6×** |
| `learn_em_emission` | 3098.7 | 470.5 | **6.6×** |
| `learn_viterbi_emission` | 2515.1 | 41.5 | **60.7×** |

Speedups grow substantially vs the 40k run — scan depth scales linearly with T, so the benefit of splitting compounds. `learn_viterbi_emission` reaches 60.7× because both its forward pass and backtrace are fully split, with no unsplit sequential bottleneck. `learn_em_transition` again lags (3.6×) for the same reason as before: its fused backward+counts scan is not split and now dominates at ~1.4s.

---

## Optimizations

**Transition scatter (HE + SE `learn_viterbi_transition`)**
`__update_transition_counts_mp`: replaced a `jax.lax.scan` that incremented `counts[a, i, j]` one timestep at a time with a single `.at[actions[:-1], states[:-1], states[1:]].add(1)`. Valid because Viterbi states are fully decoded before counting, so there is no sequential dependency.

**obs_liks precompute (HE `learn_em_emission`, `learn_viterbi_emission`; SE all methods)**
`__forward_emission`, `__backward_emission`, `__forward_emission_mp` (HE); `__forward`, `__backward`, `__forward_mp`, `__update_transition_counts` (SE): replaced T sequential column-gathers `emission_matrix[:, obs[t]]` (HE) or matvecs `emission_matrix @ obs[t]` (SE) with a single batched op computed once before the scan — `emission_matrix[:, observations].T` (HE) or `observations @ emission_matrix.T` (SE).

**Emission scatter (HE `learn_em_emission`, `learn_viterbi_emission`)**
`__update_emission_counts` and `__update_emission_counts_mp`: replaced scans over T column-add or scalar-update steps with scatter-adds — `emission_counts.at[:, obs].add(gamma.T)` and `counts.at[states, obs].add(1)` respectively.

**Emission matmul (SE `learn_em_emission`)**
`__update_emission_counts`: replaced a scan of T outer-product accumulations with a single `gamma.T @ observations` matmul. Clean replacement because soft observations are already a float `[T, E]` matrix.
