"""Setup ksp rocket package."""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()

setup(
    name="rocket_util",
    version="0.2.0",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
)
