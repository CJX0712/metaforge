"""End-to-end pipeline: build scenarios, train, benchmark, report."""

from __future__ import annotations

import json
import os
from typing import Optional

from ..data.samplers import SinusoidSampler, blob_preset
from ..eval.benchmark import run_benchmark
from ..metal.registry import default_algorithms


class MetaPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_scenarios(self, domain: str):
        cfg = self.cfg
        if domain == "cls":
            scen = []
            for lv in ["easy", "medium", "hard"]:
                tr = blob_preset(
                    lv,
                    n_way=cfg.n_way,
                    k_shot=cfg.k_shot,
                    q_shot=cfg.q_shot,
                    dim=cfg.dim,
                    seed=cfg.seed,
                )
                ev = blob_preset(
                    lv,
                    n_way=cfg.n_way,
                    k_shot=cfg.k_shot,
                    q_shot=cfg.q_shot,
                    dim=cfg.dim,
                    seed=cfg.seed + 999,
                )
                scen.append((lv, tr, ev))
            return scen
        tr = SinusoidSampler(n_support=cfg.reg_support, n_query=cfg.reg_query, seed=cfg.seed)
        ev = SinusoidSampler(n_support=cfg.reg_support, n_query=cfg.reg_query, seed=cfg.seed + 999)
        return [("sinusoid", tr, ev)]

    def run(self, domain: str = "cls", algorithms=None, out_path: Optional[str] = None):
        if algorithms is None:
            algorithms = default_algorithms(self.cfg, domain)
        scenarios = self.build_scenarios(domain)
        rows = run_benchmark(self.cfg, algorithms, scenarios)

        mf_reports = []
        for a in algorithms:
            if getattr(a, "name", None) == "metafuse" and hasattr(a, "_report"):
                mf_reports.append(a._report)

        report = {
            "system": "MetaForge",
            "author": "晨星",
            "domain": domain,
            "seed": self.cfg.seed,
            "config": self.cfg.to_dict(),
            "results": [
                {
                    **r.as_row(),
                    "level": r.detail["level"],
                    "higher_better": r.detail["higher_better"],
                }
                for r in rows
            ],
            "metafuse_reports": mf_reports,
        }
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return report, rows

    @staticmethod
    def format_table(report: dict) -> str:
        levels = []
        for r in report["results"]:
            if r["level"] not in levels:
                levels.append(r["level"])
        lines = [f"MetaForge benchmark — domain={report['domain']} (seed {report['seed']})"]
        lines.append("metric: " + ("accuracy up" if report["domain"] == "cls" else "rmse down"))
        for lv in levels:
            lines.append("")
            lines.append(f"== level: {lv} ==")
            hdr = f"{'algorithm':<22}{'mean':>10}{'std':>10}{'n':>6}"
            lines.append(hdr)
            rows = [r for r in report["results"] if r["level"] == lv]
            hb = rows[0]["higher_better"]
            rows.sort(key=lambda r: r["mean"], reverse=hb)
            for r in rows:
                lines.append(f"{r['algorithm']:<22}{r['mean']:>10.4f}{r['std']:>10.4f}{r['n']:>6}")
        return "\n".join(lines)
