from experiments.rl.run_l243_direct_control_teacher_collection import collector_argv


def test_frozen_l243_collection_contract():
    argv = collector_argv(
        "configs/rl/direct_control_bc_teacher_l243.yaml",
        "results/test_l243_collection",
    )

    assert argv[argv.index("--output-dir") + 1] == "results/test_l243_collection"
    assert argv[argv.index("--route-source") + 1] == "task_points"
    assert argv[argv.index("--teacher-action-mode") + 1] == "direct_control"
    assert argv[argv.index("--teacher-pose-source") + 1] == "ground_truth"
    assert argv[argv.index("--lookahead") + 1] == "0.35"
    assert argv[argv.index("--max-steps") + 1] == "1500"
    assert len(argv[argv.index("--seeds") + 1].split(",")) == 15
    assert argv.index("--output-dir") - argv.index("--configs") == 7
    assert "--allow-empty-splits" not in argv
    assert "--allow-non-ground-truth-localization" in argv
