import json
from pathlib import Path

from experiments.rl.generate_l258_tracking_curriculum import FAMILIES, generate


def test_l258_generator_is_deterministic_and_isolated(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate(a, seed=20262441)
    generate(b, seed=20262441)
    assert sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir())
    manifest = json.loads((a / "manifest.json").read_text())
    manifest_b = json.loads((b / "manifest.json").read_text())
    assert [r["family"] for r in manifest["routes"]] == [
        r["family"] for r in manifest_b["routes"]
    ]
    assert [p.read_text() for p in sorted(a.glob("*.yaml"))] == [
        p.read_text() for p in sorted(b.glob("*.yaml"))
    ]
    assert manifest["families"] == list(FAMILIES)
    assert len(manifest["routes"]) == 10
    text = "\n".join(p.read_text() for p in a.glob("*.yaml")).lower()
    for forbidden in ("s_chicane", "infinity", "sealed", "l247", "l256"):
        assert forbidden not in text
    assert all("l258_generated" in p.read_text() for p in a.glob("*.yaml"))
