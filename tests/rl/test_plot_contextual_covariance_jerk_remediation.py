import matplotlib.pyplot as plt

from experiments.rl.plot_contextual_covariance_jerk_remediation import build_figure


def _entry(mean=-0.01, low=-0.02, high=-0.001):
    output = {}
    for metric in (
        "control_jerk", "applied_control_jerk", "cross_track_rmse",
        "elapsed_s", "planner_compute_ms_mean",
    ):
        output[metric + "_delta_mean"] = mean
        output[metric + "_delta_ci95"] = [low, high]
    return output


def test_jerk_figure_has_six_populated_panels():
    l96 = {"contrasts_vs_current": {
        name: _entry() for name in (
            "hard_yaw_slew", "stronger_rate_cost", "combined"
        )
    }}
    l97 = {"contrasts": {
        name: _entry() for name in (
            "raw_contextual_vs_fixed", "smoothed_contextual_vs_raw",
            "smoothed_contextual_vs_fixed",
        )
    }}
    figure = build_figure(l96, l97)
    assert len(figure.axes) == 6
    assert all(axis.get_title(loc="left") for axis in figure.axes)
    plt.close(figure)
