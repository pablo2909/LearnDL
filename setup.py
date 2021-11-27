from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent

with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

with open(Path(BASE_DIR, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="LearnDL",
    version="0.1",
    license="MIT",
    description="Package to learn best practices in MLOps and DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Paul Jeha",
    author_email="paul.jeha@gmail.com",
    url="https://github.com/pablo2909/LearnDL",
    keywords=["machine-learning", "artificial-intelligence",],
    python_requires="==3.9.6",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[required_packages],
)
