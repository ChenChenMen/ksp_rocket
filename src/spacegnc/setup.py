"""Setup trajectory optimization problem package."""

from setuptools import setup, find_packages


# Read requirements from requirements.txt
def read_requirements(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()


setup(
    name="spacegnc",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
)
