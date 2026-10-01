"""Command-line entry point for MetaForge.

Usage (from the repo root)::

    python -m metaforge.cli --domain cls --out benchmark.json
    python metaforge/cli.py --domain reg --algorithm protonet,reptile,metafuse

Prints a per-level benchmark table and writes ``benchmark.json``.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metaforge.core.config import MetaConfig
from metaforge.metal.registry import REGISTRY, default_algorithms
from metaforge.pipeline.pipeline import MetaPipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description="MetaForge — few-shot meta-learning benchmark")
    ap.add_argument("--domain", choices=["cls", "reg", "all"], default="cls")
    ap.add_argument("--algorithm", default="all", help="comma-separated names or 'all'")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--meta-iters", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args(argv)

    overrides = {}
    if args.meta_iters is not None:
        overrides["meta_iters"] = args.meta_iters
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    cfg = MetaConfig.from_env(seed=args.seed, **overrides)

    domains = ["cls", "reg"] if args.domain == "all" else [args.domain]

    for domain in domains:
        if args.algorithm == "all":
            algs = default_algorithms(cfg, domain)
        else:
            names = [n.strip() for n in args.algorithm.split(",") if n.strip()]
            algs = [REGISTRY[n](cfg) for n in names]

        out_path = args.out
        if args.domain == "all":
            base, ext = os.path.splitext(args.out)
            out_path = f"{base}_{domain}{ext}"

        pipe = MetaPipeline(cfg)
        report, _ = pipe.run(domain=domain, algorithms=algs, out_path=out_path)
        print(pipe.format_table(report))
        if report.get("metafuse_reports"):
            print("\nMetaFuse (SMF) validation report:")
            for mfr in report["metafuse_reports"]:
                print(
                    f"  [{mfr['kind']}] use_fusion={mfr['use_fusion']} "
                    f"weights={[round(w, 3) for w in mfr['weights']]} "
                    f"member_scores={[round(s, 3) for s in mfr['member_scores']]} "
                    f"fused={round(mfr['fused_score'], 3)} best={mfr['best_member']}"
                )
        print(f"\n>> wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
