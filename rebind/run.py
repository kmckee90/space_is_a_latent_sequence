"""
run.py
------
Main experiment runner for the LIALT rebinding experiment.

Pipeline
--------
  1. Load and tokenize the training and test data.
  2. Build vocabularies (train vocab → model size K; full vocab for encoding).
  3. Train CSCG with EM + optional Viterbi refinement.
  4. Plot the learned transition graph.
  5. Evaluate on instruction-based and example-based test prompts.

Usage
-----
    uv run python rebind/run.py
    uv run python rebind/run.py --clones 40 --em 500 --vit 10 --test_n 200
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rebind.model import CSCG
from rebind.data import tokenize, build_vocab, encode, parse_test_file
from rebind.eval import eval_with_rebinding
from rebind.vis import plot_transition_graph

DELIM_TOKEN = "/"


def main() -> None:
    ap = argparse.ArgumentParser(description="LIALT rebinding experiment")
    ap.add_argument("--train_path",        default="lialt_train.txt")
    ap.add_argument("--test_path",         default="lialt_test.txt")
    ap.add_argument("--seed",              type=int,   default=0)
    ap.add_argument("--clones",            type=int,   default=30,
                    help="Clone states per token")
    ap.add_argument("--em",                type=int,   default=750,
                    help="EM (Baum-Welch) iterations")
    ap.add_argument("--vit",               type=int,   default=10,
                    help="Viterbi refinement iterations (0 to skip)")
    ap.add_argument("--pseudocount",       type=float, default=1e-6)
    ap.add_argument("--test_n",            type=int,   default=100,
                    help="Number of test prompts per condition")
    ap.add_argument("--rebind_iters",      type=int,   default=10,
                    help="Algorithm 1 EM iterations per prompt")
    ap.add_argument("--rebind_surprise_p", type=float, default=0.1,
                    help="Confidence threshold θ for Algorithm 1")
    ap.add_argument("--show_every",        type=int,   default=1,
                    help="Print rollout for every N-th prompt (0 = silent)")
    ap.add_argument("--graph_out",         default="images/trained_transition_graph.png")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    with open(args.train_path, encoding="utf-8") as f:
        train_lines = [line.rstrip("\n") for line in f if line.strip()]

    train_token_seqs = [tokenize(line) for line in train_lines]

    instr_prompts, example_prompts = parse_test_file(args.test_path, args.test_n)
    instr_prompts   = instr_prompts[:args.test_n]
    example_prompts = example_prompts[:args.test_n]

    # ------------------------------------------------------------------
    # 2. Build vocabularies
    #    train_vocab : training tokens only → model size K
    #    full vocab  : train + test; test-only tokens get ids >= K
    # ------------------------------------------------------------------
    all_test_toks = [p.prompt_tokens + p.target_tokens
                     for p in instr_prompts + example_prompts]
    train_vocab = build_vocab(train_token_seqs)
    full_vocab  = build_vocab(train_token_seqs + all_test_toks)
    inv_vocab   = {v: k for k, v in full_vocab.items()}
    delimiter_id = full_vocab.get(DELIM_TOKEN, -1)

    print(f"Train vocab size : {len(train_vocab)}  (K)")
    print(f"Full vocab size  : {len(full_vocab)}")
    print(f"Train sequences  : {len(train_lines)}")
    print(f"Instr prompts    : {len(instr_prompts)}")
    print(f"Example prompts  : {len(example_prompts)}")
    print(f"Delimiter '{DELIM_TOKEN}' → id {delimiter_id}")

    # ------------------------------------------------------------------
    # 3. Encode training sequences using the full vocab
    # ------------------------------------------------------------------
    train_seqs = [encode(seq, full_vocab) for seq in train_token_seqs]

    # ------------------------------------------------------------------
    # 4. Train CSCG
    # ------------------------------------------------------------------
    model = CSCG(vocab_size=len(train_vocab), clones_per_token=args.clones, seed=args.seed)

    print(f"\nTraining: {args.em} EM iters ...")
    model.fit(train_seqs, n_iters=args.em, pseudocount=args.pseudocount, verbose=True)

    if args.vit > 0:
        print(f"Viterbi refinement: {args.vit} iters ...")
        model.viterbi_train(train_seqs, n_iters=args.vit,
                            pseudocount=args.pseudocount, verbose=True)

    # ------------------------------------------------------------------
    # 5. Visualise learned transition graph
    # ------------------------------------------------------------------
    T_red, keep = model.reduced_transition_matrix(threshold=0.01)
    print(f"\nReduced graph: {len(keep)}/{model.H} states retained")

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_transition_graph(
        T_red, ax=ax, threshold=0.01,
        title=f"Trained CSCG: {len(keep)}/{model.H} states",
        node_size=100,
    )
    plt.tight_layout()
    plt.savefig(args.graph_out, dpi=150)
    plt.close(fig)
    print(f"Saved transition graph → {args.graph_out}")

    # ------------------------------------------------------------------
    # 6. Evaluate: instruction-based and example-based retrieval
    # ------------------------------------------------------------------
    eval_kwargs = dict(
        model=model,
        vocab=full_vocab,
        inv_vocab=inv_vocab,
        delimiter_id=delimiter_id,
        eps=args.pseudocount,
        p_surprise=args.rebind_surprise_p,
        n_iters=args.rebind_iters,
        show_every=args.show_every,
    )

    acc_instr = eval_with_rebinding(
        prompts=instr_prompts,
        label="Instruction-based retrieval",
        **eval_kwargs,
    )

    acc_example = eval_with_rebinding(
        prompts=example_prompts,
        label="Example-based retrieval",
        **eval_kwargs,
    )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"  Instruction-based exact acc : {acc_instr:.3f}")
    print(f"  Example-based exact acc     : {acc_example:.3f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
