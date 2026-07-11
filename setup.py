"""Compatibility shim for the older pip shipped with Ubuntu 20.04.

Modern installers read ``pyproject.toml``.  This file only enables editable
installs in environments whose pip predates PEP 660.
"""

from setuptools import find_packages, setup


setup(
    name="mobile-robot-mppi",
    version="0.2.0",
    description="Research platform for MPPI, residual dynamics, and MuJoCo",
    package_dir={"": "src"},
    packages=find_packages("src", include=("mobile_robot_mppi", "mobile_robot_mppi.*")),
    python_requires=">=3.8",
    install_requires=("numpy>=1.22,<2.0", "PyYAML>=5.3"),
    entry_points={"console_scripts": ("mppi-sim=mobile_robot_mppi.cli.simulate:main",)},
)
