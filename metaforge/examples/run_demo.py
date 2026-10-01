"""End-to-end MetaForge demo.

Generates synthetic few-shot tasks, runs the full cross-algorithm benchmark
(classification: easy/medium/hard; regression: sinusoid), prints tables and
writes ``benchmark.json`` at the repo root.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from metaforge.core.config import MetaConfig  # noqa: E402
from metaforge.pipeline.pipeline import MetaPipeline  # noqa: E402


def main():
    cfg = MetaConfig(seed=20261002, meta_iters=400, eval_episodes=200)
    pipe = MetaPipeline(cfg)

    print("#" * 70)
    print("# MetaForge — 元学习 / Few-Shot 基准演示")
    print("# Author: 晨星")
    print("#" * 70)

    for domain in ["cls", "reg"]:
        report, _ = pipe.run(
            domain=domain, out_path="benchmark.json" if domain == "cls" else "benchmark_reg.json"
        )
        print("\n" + pipe.format_table(report))
        if report.get("metafuse_reports"):
            for mfr in report["metafuse_reports"]:
                print(
                    f"  MetaFuse use_fusion={mfr['use_fusion']} "
                    f"weights={[round(w, 3) for w in mfr['weights']]} "
                    f"fused={round(mfr['fused_score'], 3)} best={mfr['best_member']}"
                )

    print("\nDone. Artifacts: benchmark.json (cls), benchmark_reg.json (reg)")


if __name__ == "__main__":
    main()
