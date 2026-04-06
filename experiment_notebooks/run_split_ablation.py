#!/usr/bin/env python3
"""Run split ablation: compare num_splits = no-split, 2, 4, 8, 16.

Trains SE and HE models at each split level on identical data and initial
weights. Reports per-EM-iteration wall-clock time and saves recovered-graph
images. Generates a markdown report.
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

from cscg import cscg_se, cscg_se_split_base
from cscg import cscg_he, cscg_he_split_base
from cscg import utils

plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N = 6                  # grid size
LENGTH = 120000        # raw walk length
SPLIT_VALUES = [2, 4, 8, 16]
N_TIMING_ITER = 15     # EM iters to time (after 1 warmup)
N_EM_ITER = 150
N_VIT_ITER = 10
PSEUDOCOUNT = 5e-4
SEED = 0

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), 'split_ablation_results'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Grid + data
# ---------------------------------------------------------------------------
def build_grid(n):
    grid = np.zeros((n, n), dtype=int)
    grid[0, 1:-1] = 1
    grid[-1, 1:-1] = 2
    grid[1:-1, 0] = 3
    grid[1:-1, -1] = 4
    grid[0, 0] = 5
    grid[-1, 0] = 6
    grid[0, -1] = 7
    grid[-1, -1] = 8
    return grid

def generate_random_walk(grid, length, seed=0):
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    r, c = n // 2, n // 2
    obs = np.zeros(length, dtype=int)
    act = np.zeros(length, dtype=int)
    pos = np.zeros((length, 2), dtype=int)
    obs[0] = grid[r, c]
    pos[0] = (r, c)
    for t in range(length - 1):
        a = int(rng.integers(0, 4))
        dr, dc = deltas[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            r, c = nr, nc
        act[t] = a
        obs[t + 1] = grid[r, c]
        pos[t + 1] = (r, c)
    return act, obs, pos

grid = build_grid(N)
obs_names = [
    'Interior', 'Top edge', 'Bottom edge', 'Left edge', 'Right edge',
    'TL corner', 'BL corner', 'TR corner', 'BR corner',
]
cmap_grid = plt.cm.get_cmap('tab10', 9)

a_full, x_full, pos_full = generate_random_walk(grid, LENGTH, seed=42)

n_obs = int(grid.max()) + 1
n_clones = 3 * np.array([(grid == i).sum() for i in range(n_obs)], dtype=int)
print(f'n_clones: {n_clones}, total states: {n_clones.sum()}')

n_devices = jax.device_count()
max_splits = max(SPLIT_VALUES)
T = (LENGTH // (n_devices * max_splits)) * (n_devices * max_splits)
x_train = x_full[:T]
a_train = a_full[:T]
pos_train = pos_full[:T]
print(f'T={T:,}, devices={n_devices}, max_splits={max_splits}')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_and_build_pos(model):
    states = np.array(model.decode(observations=x_train, actions=a_train)[1])
    state_pos_counts = {}
    for t, s in enumerate(states):
        rc = (int(pos_train[t, 0]), int(pos_train[t, 1]))
        if s not in state_pos_counts:
            state_pos_counts[s] = {}
        state_pos_counts[s][rc] = state_pos_counts[s].get(rc, 0) + 1
    state_to_pos = {s: max(cnt, key=cnt.get) for s, cnt in state_pos_counts.items()}
    unique_states = np.unique(states)
    pos_graph = {
        i: (state_to_pos[s][1], -state_to_pos[s][0])
        for i, s in enumerate(unique_states)
    }
    return states, pos_graph


def run_model(label, model, init_counts):
    """Time, train, and decode a model. Returns result dict."""
    print(f'\n--- {label} ---')

    # --- Timing (reset → warmup → time N_TIMING_ITER iters) ---
    model.set_counts_matrix(init_counts)
    model.learn_em_transition(
        observations=x_train, actions=a_train, n_iter=1, term_early=False
    )
    t0 = time.perf_counter()
    model.learn_em_transition(
        observations=x_train, actions=a_train, n_iter=N_TIMING_ITER, term_early=False
    )
    t1 = time.perf_counter()
    sec_per_em_iter = (t1 - t0) / N_TIMING_ITER
    print(f'  EM sec/iter (steady-state): {sec_per_em_iter:.3f}s')

    # --- Full training from clean initial state ---
    model.set_counts_matrix(init_counts)
    conv_em = model.learn_em_transition(
        observations=x_train, actions=a_train, n_iter=N_EM_ITER, term_early=True
    )
    conv_vit = model.learn_viterbi_transition(
        observations=x_train, actions=a_train, n_iter=N_VIT_ITER
    )
    print(f'  EM iters: {len(conv_em)}, final bps: {conv_em[-1]:.4f}')
    print(f'  Vit iters: {len(conv_vit)}, final bps: {conv_vit[-1]:.4f}')

    states, pos_graph = decode_and_build_pos(model)
    n_unique = len(np.unique(states))
    print(f'  Decoded states: {n_unique} (expected {N*N})')

    return {
        'model': model,
        'conv_em': conv_em,
        'conv_vit': conv_vit,
        'states': states,
        'pos_graph': pos_graph,
        'sec_per_em_iter': sec_per_em_iter,
        'n_unique_states': n_unique,
    }


def plot_graph_on_ax(ax, model, states, pos_graph, title):
    ax.imshow(
        grid, cmap=cmap_grid, vmin=-0.5, vmax=8.5, alpha=0.15,
        extent=[-0.5, N - 0.5, -(N - 0.5), 0.5],
    )
    utils.plot_graph(
        model.counts_matrix, states, n_clones, x_train.max(),
        ax=ax, pos=pos_graph, node_size=300, threshold=0.0, cmap='tab10',
    )
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-(N - 0.5), 0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


# ---------------------------------------------------------------------------
# SE experiment
# ---------------------------------------------------------------------------
print('\n' + '=' * 60)
print('SE MODELS')
print('=' * 60)

se_model_baseline = cscg_se.CSCG(
    n_clones=n_clones, pseudocount=PSEUDOCOUNT, n_actions=4,
    seed=SEED, batched=True, use_bfloat16=False,
)
se_init_counts = se_model_baseline.counts_matrix.copy()

se_results = {}
se_results['no-split'] = run_model('SE no-split', se_model_baseline, se_init_counts)

for ns in SPLIT_VALUES:
    m = cscg_se_split_base.CSCG(
        n_clones=n_clones, pseudocount=PSEUDOCOUNT, n_actions=4,
        seed=SEED, batched=True, use_bfloat16=False, num_splits=ns,
    )
    assert np.allclose(m.counts_matrix, se_init_counts), 'Initial counts mismatch!'
    se_results[f'split-{ns}'] = run_model(f'SE split={ns}', m, se_init_counts)

# ---------------------------------------------------------------------------
# HE experiment
# ---------------------------------------------------------------------------
print('\n' + '=' * 60)
print('HE MODELS')
print('=' * 60)

he_model_baseline = cscg_he.CSCG(
    n_clones=n_clones, pseudocount=PSEUDOCOUNT, n_actions=4,
    seed=SEED, batched=True, use_bfloat16=False,
)
he_init_counts = he_model_baseline.counts_matrix.copy()

he_results = {}
he_results['no-split'] = run_model('HE no-split', he_model_baseline, he_init_counts)

for ns in SPLIT_VALUES:
    m = cscg_he_split_base.CSCG(
        n_clones=n_clones, pseudocount=PSEUDOCOUNT, n_actions=4,
        seed=SEED, batched=True, use_bfloat16=False, num_splits=ns,
    )
    assert np.allclose(m.counts_matrix, he_init_counts), 'Initial counts mismatch!'
    he_results[f'split-{ns}'] = run_model(f'HE split={ns}', m, he_init_counts)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
all_split_keys = ['no-split'] + [f'split-{ns}' for ns in SPLIT_VALUES]
n_cols = len(all_split_keys)

def make_graph_figure(results, model_type):
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5))
    for i, key in enumerate(all_split_keys):
        res = results[key]
        label = key.replace('-', '=').replace('no=split', 'no split')
        sec = res['sec_per_em_iter']
        n_unique = res['n_unique_states']
        title = f'{model_type} {label}\n{sec:.3f}s/iter  |  {n_unique} states'
        plot_graph_on_ax(axes[i], res['model'], res['states'], res['pos_graph'], title)
    patches = [
        mpatches.Patch(color=cmap_grid(i / 8), label=f'{i}: {obs_names[i]}')
        for i in range(9)
    ]
    axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    fig.suptitle(
        f'{model_type}: recovered graphs by num_splits  '
        f'(same data, same init, {N_EM_ITER} EM + {N_VIT_ITER} Viterbi iters)',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{model_type.lower()}_graphs.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path

se_graph_path = make_graph_figure(se_results, 'SE')
he_graph_path = make_graph_figure(he_results, 'HE')

# Convergence plot
def make_convergence_figure(results, model_type):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_cols))
    for i, key in enumerate(all_split_keys):
        res = results[key]
        label = key.replace('-', '=').replace('no=split', 'no split')
        axes[0].plot(res['conv_em'], color=colors[i], label=label)
        axes[1].plot(res['conv_vit'], color=colors[i], label=label)
    axes[0].set_xlabel('EM iteration')
    axes[0].set_ylabel('bps')
    axes[0].set_title(f'{model_type} EM convergence')
    axes[0].legend()
    axes[1].set_xlabel('Viterbi iteration')
    axes[1].set_ylabel('bps')
    axes[1].set_title(f'{model_type} Viterbi convergence')
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{model_type.lower()}_convergence.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path

se_conv_path = make_convergence_figure(se_results, 'SE')
he_conv_path = make_convergence_figure(he_results, 'HE')

# Speed summary plot
def make_speed_figure(se_results, he_results):
    fig, ax = plt.subplots(figsize=(8, 4))
    x_labels = all_split_keys
    x_pos = np.arange(len(x_labels))
    se_times = [se_results[k]['sec_per_em_iter'] for k in all_split_keys]
    he_times = [he_results[k]['sec_per_em_iter'] for k in all_split_keys]
    se_base = se_times[0]
    he_base = he_times[0]

    ax.bar(x_pos - 0.2, [t / se_base for t in se_times], 0.35,
           label='SE (relative to no-split)', color='tab:blue', alpha=0.8)
    ax.bar(x_pos + 0.2, [t / he_base for t in he_times], 0.35,
           label='HE (relative to no-split)', color='tab:red', alpha=0.8)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([k.replace('-', '=').replace('no=split', 'no split') for k in x_labels])
    ax.set_ylabel('Relative time per EM iter\n(lower = faster)')
    ax.set_title('Speed relative to no-split baseline')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'speed_comparison.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path

speed_path = make_speed_figure(se_results, he_results)

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def fmt_speedup(results, key, base_key='no-split'):
    base = results[base_key]['sec_per_em_iter']
    t = results[key]['sec_per_em_iter']
    return f'{t:.3f}s ({base/t:.2f}×)'

def make_table(results, model_type):
    rows = []
    base_time = results['no-split']['sec_per_em_iter']
    for key in all_split_keys:
        r = results[key]
        t = r['sec_per_em_iter']
        speedup = base_time / t
        label = key.replace('-', '=').replace('no=split', 'no split')
        rows.append(
            f'| {label} | {r["n_unique_states"]} | {len(r["conv_em"])} | '
            f'{r["conv_em"][-1]:.4f} | {len(r["conv_vit"])} | '
            f'{r["conv_vit"][-1]:.4f} | {t:.3f}s | {speedup:.2f}× |'
        )
    header = (
        f'| Model | Decoded states | EM iters | EM final bps | '
        f'Vit iters | Vit final bps | sec/EM iter | Speedup vs no-split |\n'
        f'|---|---|---|---|---|---|---|---|'
    )
    return header + '\n' + '\n'.join(rows)

report = textwrap.dedent(f"""\
# Split training ablation: num_splits ∈ {{no-split, 2, 4, 8, 16}}

**Setup:** {N}×{N} colored grid world, T={T:,} steps, seed={SEED}.
All models within each type (SE/HE) start from the same initial counts matrix.
The only difference is the number of sequence splits used during EM/Viterbi training.
EM: up to {N_EM_ITER} iters (early stop). Viterbi: {N_VIT_ITER} iters.
Timing: steady-state EM wall-clock (after 1 JIT-warmup iter), averaged over {N_TIMING_ITER} iterations.

Ground truth: {N}×{N} = {N*N} grid cells, {n_clones.sum()} total clones.

---

## Soft evidence (SE) results

### Recovered graphs

![SE graphs](split_ablation_results/se_graphs.png)

### Convergence

![SE convergence](split_ablation_results/se_convergence.png)

### Summary table

{make_table(se_results, 'SE')}

---

## Hard evidence (HE) results

### Recovered graphs

![HE graphs](split_ablation_results/he_graphs.png)

### Convergence

![HE convergence](split_ablation_results/he_convergence.png)

### Summary table

{make_table(he_results, 'HE')}

---

## Speed comparison

![Speed comparison](split_ablation_results/speed_comparison.png)

### SE per-iteration timing

| num_splits | sec/EM iter | speedup |
|---|---|---|
""")

base_se = se_results['no-split']['sec_per_em_iter']
for key in all_split_keys:
    t = se_results[key]['sec_per_em_iter']
    label = key.replace('-', '=').replace('no=split', 'no split')
    report += f'| {label} | {t:.3f}s | {base_se/t:.2f}× |\n'

report += '\n### HE per-iteration timing\n\n'
report += '| num_splits | sec/EM iter | speedup |\n|---|---|---|\n'

base_he = he_results['no-split']['sec_per_em_iter']
for key in all_split_keys:
    t = he_results[key]['sec_per_em_iter']
    label = key.replace('-', '=').replace('no=split', 'no split')
    report += f'| {label} | {t:.3f}s | {base_he/t:.2f}× |\n'

report += textwrap.dedent("""
---

## Observations

- **Graph quality:** Assess whether increasing num_splits degrades the recovered graph topology.
  Fewer unique decoded states (closer to the true grid cell count) is better.
- **Speed:** Each doubling of num_splits roughly halves the sequential scan length for the
  forward/backward passes, but the transition-count update scan is NOT split (it has a genuine
  Markov dependency), so total speedup is sub-linear in num_splits.
- **Diminishing returns:** Beyond a certain num_splits the per-iteration time flattens because
  the non-split operations dominate.
""")

report_path = os.path.join(
    os.path.dirname(__file__), 'split_ablation_report.md'
)
with open(report_path, 'w') as f:
    f.write(report)

print(f'\nReport written to {report_path}')
print('\nDone.')
