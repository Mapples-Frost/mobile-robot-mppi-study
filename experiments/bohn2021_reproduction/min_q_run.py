"""Use the existing paper-grid trainer with one altered actor objective."""
import run
from min_q_sac import SOURCE, load_min_q_sac, source_sha256


if __name__ == "__main__":
    original_imports = run.imports
    MinQSAC = load_min_q_sac()
    original_write = run.write

    def modified_imports():
        env, _, policy = original_imports()
        return env, MinQSAC, policy

    def annotated_write(path, data):
        if path.name == "manifest.json":
            data["method_extension"] = {
                "name": "actor_min_q",
                "only_algorithm_change": "actor objective uses min(Q1,Q2) instead of Q1",
                "source": str(SOURCE),
                "source_sha256": source_sha256(),
                "strict_paper_reproduction": False,
            }
        elif path.name == "completed.json" and path.parent.name.startswith(("vehicle_min_q_", "pendulum_min_q_")):
            data["reproduction_level"] = "method extension: author SAC actor uses min(Q1,Q2)"
        original_write(path, data)

    run.imports = modified_imports
    run.write = annotated_write
    run.main()
