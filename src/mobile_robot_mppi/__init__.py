"""Composable research platform for mobile-robot MPPI experiments.

The package deliberately separates the environment true plant from the
prediction dynamics used inside MPPI.  MuJoCo, ROS, and Torch are optional
integration dependencies and are never imported by the package root.
"""

__version__ = "0.2.0"
