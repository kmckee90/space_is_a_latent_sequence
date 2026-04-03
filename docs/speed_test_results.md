# Speed Test Results — Original vs Optimized

**Config:** 20 emissions · 10 clones/obs (200 states) · 4 actions · SEQ_LEN=10,000 · 6 bench iters

---

## HE (Hard Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | transition scatter | 4.3 | 2.2 | **1.94×** | 39.3 | 28.4 | **1.39×** |
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
| `learn_viterbi_transition` | transition scatter | 85.9 | 84.8 | 1.01× | 101.0 | 74.8 | **1.35×** |
| `learn_em_transition` | *not optimized (control)* | 89.5 | 86.5 | 1.04× | 73.2 | 73.1 | 1.00× |
| `learn_em_emission` | scatter + obs_liks precompute | 15118.1 | 13642.9 | **1.11×** | 432.0 | 379.5 | **1.14×** |
| `learn_viterbi_emission` | scatter + obs_liks precompute | 9310.3 | 8193.7 | **1.14×** | 201.4 | 169.8 | **1.19×** |

### SE (Soft Evidence)

| Method | Optimization | CPU orig (ms) | CPU opt (ms) | CPU speedup | GPU orig (ms) | GPU opt (ms) | GPU speedup |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `learn_viterbi_transition` | scatter + obs_liks precompute | 11243.5 | 11837.2 | 0.95× | 113.8 | 83.2 | **1.37×** |
| `learn_em_transition` | obs_liks precompute in scan | 55068.3 | 55970.7 | 0.98× | 94.4 | 90.4 | 1.04× |
| `learn_em_emission` | matmul + obs_liks precompute | 13787.1 | 10369.4 | **1.33×** | 427.0 | 384.3 | **1.11×** |
