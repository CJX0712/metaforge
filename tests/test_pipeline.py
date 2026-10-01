import json

from metaforge.core.config import MetaConfig
from metaforge.metal.registry import default_algorithms
from metaforge.pipeline.pipeline import MetaPipeline


def _run(domain):
    cfg = MetaConfig(seed=1, meta_iters=80, eval_episodes=30)
    pipe = MetaPipeline(cfg)
    report, _ = pipe.run(domain=domain, algorithms=default_algorithms(cfg, domain), out_path=None)
    return report


def test_pipeline_runs_and_reports():
    report = _run("cls")
    assert report["system"] == "MetaForge"
    assert len(report["results"]) > 0
    for r in report["results"]:
        assert r["n"] == 30
        assert r["mean"] == r["mean"]  # not nan (sanity)


def test_pipeline_deterministic():
    r1 = _run("cls")
    r2 = _run("cls")
    m1 = {f"{x['algorithm']}/{x['level']}": x["mean"] for x in r1["results"]}
    m2 = {f"{x['algorithm']}/{x['level']}": x["mean"] for x in r2["results"]}
    for k in m1:
        assert abs(m1[k] - m2[k]) < 1e-9, f"non-deterministic for {k}"


def test_pipeline_writes_json(tmp_path):
    out = tmp_path / "benchmark.json"
    cfg = MetaConfig(seed=1, meta_iters=60, eval_episodes=20)
    pipe = MetaPipeline(cfg)
    pipe.run(domain="reg", out_path=str(out))
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["domain"] == "reg"
    assert "results" in data
