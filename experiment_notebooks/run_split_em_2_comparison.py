#!/usr/bin/env python3
"""Compare no-split baseline vs split_em_2 for HE and SE models.

Tests num_splits in [8, 16, 32, 64].  Appends a new section to
split_ablation_report.md with timing tables, graph figures, and convergence
curves.
"""

import os
import sys
import time
import textwrap

import numpy as np
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '/data/users/kevin/space_is_a_latent_sequence')

from cscg import cscg_se, cscg_se_split_em_2
from cscg import cscg_he, cscg_he_split_em_2
from cscg import utils

plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N = 6
LENGTH = 120000
SPLIT_VALUES = [8, 16, 32, 64]
N_TIMING_ITER = 15
N_EM_ITER = 150
N_VIT_ITER = 10
PSEUDOCOUNT = 5e-4
SEED = 0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'split_ablation_results')
FIG_DIR = os.path.join(RESULTS_DIR, 'split_em_2')
os.makedirs(FIG_DIR, exist_ok=True)
REPORT_PATH = os.path.join(os.path.dirname(__file__), 'split_ablation_report.md')

# ---------------------------------------------------------------------------
# Grid + data
# ---------------------------------------------------------------------------
def build_grid(n):
    grid = np.zeros((n, n), dtype=int)
    grid[0, 1:-1] = 1;  grid[-1, 1:-1] = 2
    grid[1:-1, 0] = 3;  grid[1:-1, -1] = 4
    grid[0, 0] = 5;  grid[-1, 0] = 6
    grid[0, -1] = 7;  grid[-1, -1] = 8
    return grid

def generate_random_walk(grid, length, seed=0):
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    r, c = n // 2, n // 2
    obs = np.zeros(length, dtype=int)
    act = np.zeros(length, dtype=int)
    pos = np.zeros((length, 2), dtype=int)
    obs[0] = grid[r, c];  pos[0] = (r, c)
    for t in range(length - 1):
        a = int(rng.integers(0, 4))
        dr, dc = deltas[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            r, c = nr, nc
        act[t] = a;  obs[t + 1] = grid[r, c];  pos[t + 1] = (r, c)
    return act, obs, pos

grid = build_grid(N)
obs_names = ['Interior', 'Top edge', 'Bottom edge', 'Left edge', 'Right edge',
             'TL corner', 'BL corner', 'TR corner', 'BR corner']
cmap_grid = plt.cm.get_cmap('tab10', 9)

a_full, x_full, pos_full = generate_random_walk(grid, LENGTH, seed=42)
n_obs = int(grid.max()) + 1
n_clones = 3 * np.array([(grid == i).sum() for i in range(n_obs)], dtype=int)

n_devices = jax.device_count()
max_splits = max(SPLIT_VALUES)
T = (LENGTH // (n_devices * max_splits)) * (n_devices * max_splits)
x_train = x_full[:T];  a_train = a_full[:T];  pos_train = pos_full[:T]
# One-hot for SE
x_oh = np.eye(n_obs, dtype=np.float32)[x_train]

print(f'T={T:,}, devices={n_devices}, states={n_clones.sum()}, '
      f'splits={SPLIT_VALUES}')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_and_build_pos_he(model):
    states = np.array(model.decode(observations=x_train, actions=a_train)[1])
    spc = {}
    for t, s in enumerate(states):
        rc = (int(pos_train[t, 0]), int(pos_train[t, 1]))
        spc.setdefault(s, {})
        spc[s][rc] = spc[s].get(rc, 0) + 1
    s2p = {s: max(cnt, key=cnt.get) for s, cnt in spc.items()}
    us = np.unique(states)
    pg = {i: (s2p[s][1], -s2p[s][0]) for i, s in enumerate(us)}
    return states, pg

def decode_and_build_pos_se(model):
    states = np.array(model.decode(observations=x_oh, actions=a_train)[1])
    spc = {}
    for t, s in enumerate(states):
        rc = (int(pos_train[t, 0]), int(pos_train[t, 1]))
        spc.setdefault(s, {})
        spc[s][rc] = spc[s].get(rc, 0) + 1
    s2p = {s: max(cnt, key=cnt.get) for s, cnt in spc.items()}
    us = np.unique(states)
    pg = {i: (s2p[s][1], -s2p[s][0]) for i, s in enumerate(us)}
    return states, pg

def run_he_model(label, model, init_counts):
    print(f'  [{label}]')
    # Warmup + timing
    model.set_counts_matrix(init_counts)
    model.learn_em_transition(observations=x_train, actions=a_train,
                              n_iter=1, term_early=False)
    model.set_counts_matrix(init_counts)
    t0 = time.perf_counter()
    model.learn_em_transition(observations=x_train, actions=a_train,
                              n_iter=N_TIMING_ITER, term_early=False)
    em_sec = (time.perf_counter() - t0) / N_TIMING_ITER

    # Full training
    model.set_counts_matrix(init_counts)
    conv_em = model.learn_em_transition(observations=x_train, actions=a_train,
                                        n_iter=N_EM_ITER, term_early=True)
    conv_vit = model.learn_viterbi_transition(observations=x_train, actions=a_train,
                                              n_iter=N_VIT_ITER)
    states, pg = decode_and_build_pos_he(model)
    n_unique = len(np.unique(states))
    print(f'    EM: {len(conv_em)} iters → {conv_em[-1]:.4f} bps  ({em_sec:.3f}s/iter)')
    print(f'    Vit: {len(conv_vit)} iters → {conv_vit[-1]:.4f} bps  |  '
          f'decoded states: {n_unique}')
    return dict(conv_em=conv_em, conv_vit=conv_vit, states=states, pos_graph=pg,
                em_sec=em_sec, n_unique=n_unique, model=model)

def run_se_model(label, model, init_counts):
    print(f'  [{label}]')
    model.set_counts_matrix(init_counts)
    model.learn_em_transition(observations=x_oh, actions=a_train,
                              n_iter=1, term_early=False)
    model.set_counts_matrix(init_counts)
    t0 = time.perf_counter()
    model.learn_em_transition(observations=x_oh, actions=a_train,
                              n_iter=N_TIMING_ITER, term_early=False)
    em_sec = (time.perf_counter() - t0) / N_TIMING_ITER

    model.set_counts_matrix(init_counts)
    conv_em = model.learn_em_transition(observations=x_oh, actions=a_train,
                                        n_iter=N_EM_ITER, term_early=True)
    conv_vit = model.learn_viterbi_transition(observations=x_oh, actions=a_train,
                                              n_iter=N_VIT_ITER)
    states, pg = decode_and_build_pos_se(model)
    n_unique = len(np.unique(states))
    print(f'    EM: {len(conv_em)} iters → {conv_em[-1]:.4f} bps  ({em_sec:.3f}s/iter)')
    print(f'    Vit: {len(conv_vit)} iters → {conv_vit[-1]:.4f} bps  |  '
          f'decoded states: {n_unique}')
    return dict(conv_em=conv_em, conv_vit=conv_vit, states=states, pos_graph=pg,
                em_sec=em_sec, n_unique=n_unique, model=model)

def plot_graph_ax(ax, model, states, pg, title):
    ax.imshow(grid, cmap=cmap_grid, vmin=-0.5, vmax=8.5, alpha=0.15,
              extent=[-0.5, N-0.5, -(N-0.5), 0.5])
    utils.plot_graph(model.counts_matrix, states, n_clones, x_train.max(),
                     ax=ax, pos=pg, node_size=300, threshold=0.0, cmap='tab10')
    ax.set_xlim(-0.5, N-0.5);  ax.set_ylim(-(N-0.5), 0.5)
    ax.set_title(title, fontsize=9);  ax.set_xticks([]);  ax.set_yticks([])
    ax.set_aspect('equal')

patches = [mpatches.Patch(color=cmap_grid(i/8), label=f'{i}: {obs_names[i]}')
           for i in range(9)]

# ---------------------------------------------------------------------------
# Run HE experiments
# ---------------------------------------------------------------------------
print('\n' + '='*60 + '\nHE experiments\n' + '='*60)

he_base_model = cscg_he.CSCG(n_clones=n_clones, pseudocount=PSEUDOCOUNT,
                               n_actions=4, seed=SEED, batched=True)
he_init = he_base_model.counts_matrix.copy()

print('\n[HE no-split baseline]')
he_results = {'no-split': run_he_model('HE no-split', he_base_model, he_init)}
he_base_sec = he_results['no-split']['em_sec']

for ns in SPLIT_VALUES:
    m = cscg_he_split_em_2.CSCG(n_clones=n_clones, pseudocount=PSEUDOCOUNT,
                                  n_actions=4, seed=SEED, batched=True,
                                  num_splits=ns)
    assert np.allclose(m.counts_matrix, he_init), "counts mismatch"
    he_results[ns] = run_he_model(f'HE split_em_2 ns={ns}', m, he_init)

# HE graphs figure
n_cols = 1 + len(SPLIT_VALUES)
fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4.5))
for col, key in enumerate(['no-split'] + SPLIT_VALUES):
    res = he_results[key]
    if key == 'no-split':
        title = f'HE no-split\n{res["em_sec"]:.3f}s/iter | {res["n_unique"]} states'
    else:
        speedup = he_base_sec / res['em_sec']
        title = (f'HE split_em_2 ns={key}\n'
                 f'{res["em_sec"]:.3f}s/iter ({speedup:.1f}×) | {res["n_unique"]} states')
    plot_graph_ax(axes[col], res['model'], res['states'], res['pos_graph'], title)
axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
fig.suptitle('HE: no-split vs split_em_2 — recovered graphs', fontsize=13)
plt.tight_layout()
he_graphs_path = os.path.join(FIG_DIR, 'he_graphs.png')
fig.savefig(he_graphs_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {he_graphs_path}')

# HE convergence figure
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for key, res in he_results.items():
    lbl = 'no-split' if key == 'no-split' else f'split_em_2 ns={key}'
    axes[0].plot(res['conv_em'], label=lbl)
    axes[1].plot(res['conv_vit'], label=lbl)
axes[0].set_title('HE EM convergence'); axes[0].set_xlabel('iteration'); axes[0].set_ylabel('bps')
axes[1].set_title('HE Viterbi convergence'); axes[1].set_xlabel('iteration')
for ax in axes: ax.legend(fontsize=9)
fig.suptitle('HE convergence: no-split vs split_em_2', fontsize=13)
plt.tight_layout()
he_conv_path = os.path.join(FIG_DIR, 'he_convergence.png')
fig.savefig(he_conv_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {he_conv_path}')

# ---------------------------------------------------------------------------
# Run SE experiments
# ---------------------------------------------------------------------------
print('\n' + '='*60 + '\nSE experiments\n' + '='*60)

se_base_model = cscg_se.CSCG(n_clones=n_clones, pseudocount=PSEUDOCOUNT,
                               n_actions=4, seed=SEED, batched=True)
se_init = se_base_model.counts_matrix.copy()

print('\n[SE no-split baseline]')
se_results = {'no-split': run_se_model('SE no-split', se_base_model, se_init)}
se_base_sec = se_results['no-split']['em_sec']

for ns in SPLIT_VALUES:
    m = cscg_se_split_em_2.CSCG(n_clones=n_clones, pseudocount=PSEUDOCOUNT,
                                  n_actions=4, seed=SEED, batched=True,
                                  num_splits=ns)
    assert np.allclose(m.counts_matrix, se_init), "counts mismatch"
    se_results[ns] = run_se_model(f'SE split_em_2 ns={ns}', m, se_init)

# SE graphs figure
fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4.5))
for col, key in enumerate(['no-split'] + SPLIT_VALUES):
    res = se_results[key]
    if key == 'no-split':
        title = f'SE no-split\n{res["em_sec"]:.3f}s/iter | {res["n_unique"]} states'
    else:
        speedup = se_base_sec / res['em_sec']
        title = (f'SE split_em_2 ns={key}\n'
                 f'{res["em_sec"]:.3f}s/iter ({speedup:.1f}×) | {res["n_unique"]} states')
    plot_graph_ax(axes[col], res['model'], res['states'], res['pos_graph'], title)
axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
fig.suptitle('SE: no-split vs split_em_2 — recovered graphs', fontsize=13)
plt.tight_layout()
se_graphs_path = os.path.join(FIG_DIR, 'se_graphs.png')
fig.savefig(se_graphs_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {se_graphs_path}')

# SE convergence figure
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for key, res in se_results.items():
    lbl = 'no-split' if key == 'no-split' else f'split_em_2 ns={key}'
    axes[0].plot(res['conv_em'], label=lbl)
    axes[1].plot(res['conv_vit'], label=lbl)
axes[0].set_title('SE EM convergence'); axes[0].set_xlabel('iteration'); axes[0].set_ylabel('bps')
axes[1].set_title('SE Viterbi convergence'); axes[1].set_xlabel('iteration')
for ax in axes: ax.legend(fontsize=9)
fig.suptitle('SE convergence: no-split vs split_em_2', fontsize=13)
plt.tight_layout()
se_conv_path = os.path.join(FIG_DIR, 'se_convergence.png')
fig.savefig(se_conv_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {se_conv_path}')

# ---------------------------------------------------------------------------
# Combined speed figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (model_type, results, base_sec) in zip(
        axes, [('SE', se_results, se_base_sec), ('HE', he_results, he_base_sec)]):
    xs = ['no-split'] + SPLIT_VALUES
    secs = [results[k]['em_sec'] for k in xs]
    speedups = [base_sec / s for s in secs]
    x_pos = np.arange(len(xs))
    bars = ax.bar(x_pos, speedups, color=['tab:gray'] + ['tab:blue']*len(SPLIT_VALUES))
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(k) for k in xs], fontsize=10)
    ax.set_xlabel('num_splits')
    ax.set_ylabel('Speedup vs no-split')
    ax.set_title(f'{model_type}: split_em_2 speedup')
    for bar, sp, sec in zip(bars, speedups, secs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{sp:.1f}×\n({sec:.3f}s)', ha='center', va='bottom', fontsize=9)
fig.suptitle('split_em_2 speedup over no-split baseline', fontsize=13)
plt.tight_layout()
speed_path = os.path.join(FIG_DIR, 'speed_comparison.png')
fig.savefig(speed_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {speed_path}')

# ---------------------------------------------------------------------------
# Build report section
# ---------------------------------------------------------------------------
def make_table(results, base_sec):
    rows = []
    for key in ['no-split'] + SPLIT_VALUES:
        r = results[key]
        speedup = base_sec / r['em_sec']
        label = 'no split' if key == 'no-split' else f'split_em_2 ns={key}'
        rows.append(
            f'| {label} | {r["n_unique"]} | {len(r["conv_em"])} | '
            f'{r["conv_em"][-1]:.4f} | {len(r["conv_vit"])} | '
            f'{r["conv_vit"][-1]:.4f} | {r["em_sec"]:.3f}s | {speedup:.2f}× |'
        )
    header = (
        '| Model | Decoded states | EM iters | EM final bps | '
        'Vit iters | Vit final bps | sec/EM iter | Speedup vs no-split |\n'
        '|---|---|---|---|---|---|---|---|'
    )
    return header + '\n' + '\n'.join(rows)

def make_speed_table(results, base_sec):
    rows = []
    for key in ['no-split'] + SPLIT_VALUES:
        r = results[key]
        speedup = base_sec / r['em_sec']
        label = 'no split' if key == 'no-split' else f'split_em_2 ns={key}'
        rows.append(f'| {label} | {r["em_sec"]:.3f}s | {speedup:.2f}× |')
    header = '| Model | sec/EM iter | Speedup |\n|---|---|---|'
    return header + '\n' + '\n'.join(rows)

# Relative paths for the report (report is in experiment_notebooks/)
rel = 'split_ablation_results/split_em_2'

section = textwrap.dedent(f"""

---

## split_em_2: parallelised counts accumulation (HE and SE)

**Summary of changes vs split_em:**
- `_update_transition_counts` replaced with a segment-parallel version (same
  `num_splits` as the EM forward/backward split).  Each segment runs an
  independent local scan of length `T/num_splits - 1`; local counts are summed.
- `_update_emission_counts` replaced with `gamma.T @ observations` (SE) or a
  direct scatter-add `emission_counts.at[:, obs].add(gamma.T)` (HE) — no scan.
- `_update_transition_counts_mp` (Viterbi) replaced with
  `counts.at[a[:-1], s[:-1], s[1:]].add(1.0)` — one-line scatter-add.
- `_update_emission_counts_mp` (HE Viterbi) similarly replaced.

**Setup:** Same 6×6 grid world, T={T:,} steps, seed=42.
Same initial counts matrix across all models within each type.
EM: up to {N_EM_ITER} iters (early stop). Viterbi: {N_VIT_ITER} iters.
Timing: steady-state (after 1 JIT-warmup iter), averaged over {N_TIMING_ITER} iters.
num_splits ∈ {{{', '.join(str(s) for s in SPLIT_VALUES)}}}.

---

### HE results

#### Recovered graphs

![HE graphs]({rel}/he_graphs.png)

#### Convergence

![HE convergence]({rel}/he_convergence.png)

#### Summary table

{make_table(he_results, he_base_sec)}

#### Timing

{make_speed_table(he_results, he_base_sec)}

---

### SE results

#### Recovered graphs

![SE graphs]({rel}/se_graphs.png)

#### Convergence

![SE convergence]({rel}/se_convergence.png)

#### Summary table

{make_table(se_results, se_base_sec)}

#### Timing

{make_speed_table(se_results, se_base_sec)}

---

### Speed comparison

![Speed comparison]({rel}/speed_comparison.png)
""")

with open(REPORT_PATH, 'a') as f:
    f.write(section)

print(f'\nAppended new section to {REPORT_PATH}')
print('Done.')
