from setuptools import setup, find_packages

setup(
    name="founder_endurance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "gymnasium.envs": [
            "founder_endurance/FounderEndurance-v1=founder_endurance.envs:FounderEnv",
        ]
    },
)