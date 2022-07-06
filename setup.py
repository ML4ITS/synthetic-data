from setuptools import find_packages, setup

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()


setup(
    name="synthetic-data",
    version="1.0.0",
    description="Synthetic Time-Series",
    author="Andreas Øie",
    author_email="oeie.andreas@gmail.com",
    url="https://github.com/ML4ITS/synthetic-data",
    packages=find_packages(exclude=["pyproject.toml"]),
    install_requires=requirements,
)
