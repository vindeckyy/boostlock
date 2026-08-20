from setuptools import setup, find_packages

setup(
    name="boostlock",
    version="0.2.0",
    description="Linux CPU boost manager",
    packages=find_packages(include=["boostlock*"]),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "boostlock=boostlock.cli:main",
        ],
    },
)
