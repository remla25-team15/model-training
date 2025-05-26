from setuptools import find_packages, setup

setup(
    name="team15-model-training",
    version="0.1.0",
    description="Model training package for CS4295 project",
    author="team-15",
    author_email="remlateam15@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.10",
)
