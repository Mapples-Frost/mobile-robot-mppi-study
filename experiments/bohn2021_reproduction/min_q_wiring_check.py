"""Check the paired short run before treating min-Q as a training intervention."""
import ast
import json

import numpy as np

from min_q_sac import ActorTarget, SOURCE, load_min_q_sac, source_sha256
from runtime import ART, imports
from run import weights_hash, write


OUT = ART / "results/min_q_wiring_2026-09-24"


def read(path):
    return json.loads(path.read_text())


def main():
    original_tree = ast.parse(SOURCE.read_text())
    transformed_tree = ast.parse(SOURCE.read_text())
    transformer = ActorTarget()
    transformer.visit(transformed_tree)
    assert transformer.replacements == 1
    original_nodes = list(ast.walk(original_tree))
    transformed_nodes = list(ast.walk(transformed_tree))
    assert len(original_nodes) == len(transformed_nodes)
    changed = [(left, right) for left, right in zip(original_nodes, transformed_nodes)
               if ast.dump(left) != ast.dump(right) and isinstance(left, ast.Name)]
    assert len(changed) == 1 and changed[0][0].id == "qf1_pi" and changed[0][1].id == "min_qf_pi"
    author = OUT / "author"
    min_q = OUT / "min_q"
    a, b = read(author / "completed.json"), read(min_q / "completed.json")
    am, bm = read(author / "manifest.json"), read(min_q / "manifest.json")
    assert am["initial_hash"] == bm["initial_hash"] == a["initial_hash"] == b["initial_hash"]
    assert a["steps"] == b["steps"] == 300 and a["updates"] == b["updates"] > 0
    assert read(author / "first_update.json")["cycle"] == read(min_q / "first_update.json")["cycle"]
    assert read(author / "first_update.json")["weights_changed"]
    assert read(min_q / "first_update.json")["weights_changed"]
    assert a["final_hash"] != b["final_hash"]
    assert bm["method_extension"]["source_sha256"] == source_sha256()
    assert not bm["method_extension"]["strict_paper_reproduction"]
    assert "method_extension" not in am

    _, SAC, _ = imports()
    MinQSAC = load_min_q_sac()
    original = SAC.load(str(author / "model.zip"))
    modified = MinQSAC.load(str(min_q / "model.zip"))
    inference_reload = SAC.load(str(min_q / "model.zip"))
    assert weights_hash(original) == a["final_hash"]
    assert weights_hash(modified) == weights_hash(inference_reload) == b["final_hash"]
    original_params, modified_params = original.get_parameters(), modified.get_parameters()
    actor_keys = [key for key in original_params if key.startswith("model/pi/")]
    assert actor_keys and set(original_params) == set(modified_params)
    actor_max_abs_difference = max(float(np.max(np.abs(original_params[key] - modified_params[key])))
                                   for key in actor_keys)
    assert actor_max_abs_difference > 0
    observation = np.zeros(am["observation_dim"], dtype=np.float32)
    actual = modified.predict(observation, deterministic=True)[0]
    ordinary_reload = inference_reload.predict(observation, deterministic=True)[0]
    assert np.array_equal(actual, ordinary_reload)
    for model in (original, modified, inference_reload):
        model.sess.close()
    result = {"passed": True, "task": "pendulum", "seed": 0, "steps": 300,
              "first_update_cycle": read(author / "first_update.json")["cycle"],
              "updates": a["updates"], "initial_hash": a["initial_hash"],
              "author_final_hash": a["final_hash"], "min_q_final_hash": b["final_hash"],
              "actor_parameter_keys": len(actor_keys),
              "actor_max_abs_difference": actor_max_abs_difference,
              "ast_name_replacements": len(changed),
              "reload_hashes_match": True, "inference_reload_equal": True,
              "scope": "Training/reload wiring only; no method-performance inference."}
    write(OUT / "wiring_check.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
