"""Setup mass prop package."""

from setuptools import setup, find_packages


# Read requirements from requirements.txt
def read_requirements(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()


setup(
    name="mass_prop",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": read_requirements("test-requirements.txt")},
)
