import os
from setuptools import find_packages, setup


NAME = "batteries"
DESCRIPTION = "Batteries for PyTorch experiments."
EMAIL = "ddoroshenko@star.global"
AUTHOR = "Dmytro Doroshenko"
REQUIRES_PYTHON = ">=3.6.0"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements():
    with open(os.path.join(PROJECT_ROOT, "requirements", "requirements.txt"), "r") as f:
        return f.read().splitlines()


def load_version():
    context = {}
    with open(os.path.join(PROJECT_ROOT, "batteries", "__version__.py")) as f:
        exec(f.read(), context)
    return context["__version__"]


setup(
    name=NAME,
    version=load_version(),
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests",)),
    install_requires=load_requirements(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
