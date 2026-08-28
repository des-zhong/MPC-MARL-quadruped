"""Installation metadata for the DribbleBot Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_METADATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")


setup(
    name="dribblebot-isaaclab",
    version=EXTENSION_METADATA["package"]["version"],
    description=EXTENSION_METADATA["package"]["description"],
    author=EXTENSION_METADATA["package"]["author"],
    maintainer=EXTENSION_METADATA["package"]["maintainer"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
