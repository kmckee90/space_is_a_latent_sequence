# Speed Test Results — Original vs Optimized

**Config:** 20 emissions · 10 clones/obs (200 states) · 4 actions · 6 bench iters

---

## CPU (`JAX_PLATFORM_NAME=cpu`, SEQ_LEN=80,000)

### HE (Hard Evidence)

| Method | Optimization | Original (ms) | Optimized (ms) | Speedup |
|--------|-------------|--------------|----------------|---------|
| `learn_viterbi_transition` | transition scatter | 20.6 | 17.5 | 1.18× |
| `learn_em_transition` | *not optimized (control)* | 77.9 | 76.9 | 1.01× |
| `learn_em_emission` | scatter + obs_liks precompute | 539.3 | 509.7 | 1.06× |
| `learn_viterbi_emission` | scatter + obs_liks precompute | 161.7 | 154.8 | 1.04× |

### SE (Soft Evidence)

| Method | Optimization | Original (ms) | Optimized (ms) | Speedup |
|--------|-------------|--------------|----------------|---------|
| `learn_viterbi_transition` | scatter + obs_liks precompute | 2598.3 | 2572.4 | 1.01× |
| `learn_em_transition` | obs_liks precompute in scan | 6188.6 | 6106.4 | 1.01× |
| `learn_em_emission` | matmul + obs_liks precompute | 753.3 | 548.7 | 1.37× |

---

## GPU (CUDA, 8× devices, SEQ_LEN=10,000)

### HE (Hard Evidence)

| Method | Optimization | Original (ms) | Optimized (ms) | Speedup |
|--------|-------------|--------------|----------------|---------|
| `learn_viterbi_transition` | transition scatter | 39.3 | 28.4 | 1.39× |
| `learn_em_transition` | *not optimized (control)* | 55.2 | 55.1 | 1.00× |
| `learn_em_emission` | scatter + obs_liks precompute | 64.7 | 52.9 | 1.22× |
| `learn_viterbi_emission` | scatter + obs_liks precompute | 37.8 | 32.4 | 1.17× |

### SE (Soft Evidence)

| Method | Optimization | Original (ms) | Optimized (ms) | Speedup |
|--------|-------------|--------------|----------------|---------|
| `learn_viterbi_transition` | scatter + obs_liks precompute | 40.0 | 32.2 | 1.24× |
| `learn_em_transition` | obs_liks precompute in scan | 40.2 | 39.1 | 1.03× |
| `learn_em_emission` | matmul + obs_liks precompute | 63.9 | 52.6 | 1.21× |

---

## Summary

| Method | CPU speedup | GPU speedup | Notes |
|--------|-------------|-------------|-------|
| HE `learn_viterbi_transition` | 1.18× | **1.39×** | Scatter benefits more on GPU |
| HE `learn_em_transition` | 1.01× | 1.00× | Not optimized — expected flat |
| HE `learn_em_emission` | 1.06× | **1.22×** | GPU amplifies scatter gain |
| HE `learn_viterbi_emission` | 1.04× | **1.17×** | Same pattern |
| SE `learn_viterbi_transition` | 1.01× | **1.24×** | CPU gains masked by overhead at 8× seq_len |
| SE `learn_em_transition` | 1.01× | 1.03× | Sequential scan limits gains on both |
| SE `learn_em_emission` | **1.37×** | **1.21×** | Matmul wins on both; large on CPU at long seq |
