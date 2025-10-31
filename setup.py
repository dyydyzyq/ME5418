from setuptools import setup, find_packages

setup(
    name="me5418-group20",
    version="0.1.0",
    author="Group20",
    description="MuJoCo-based Franka Panda simulation and SAC training environment for ME5418",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "mujoco==3.3.6",
        "gymnasium",
    ],
    python_requires=">=3.10",
)
