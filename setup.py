from setuptools import setup, find_packages

setup(
    name="boostlock",
    version="0.1.0",
    description="24/7 Sustained CPU Boost Clock Management System for Linux",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "boostlock=boostlock.cli:main",
        ],
    },
)
