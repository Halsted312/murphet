# setup.py  – PEP 517 fallback for editable installs (pip install -e .)
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

setup(
    name="murphet",
    version="1.1.0",          # ← same as pyproject
    description="A Bayesian time‑series model for churn rates with changepoints and seasonality",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Stephen Murphy",
    author_email="stephenjmurph@gmail.com",
    url="https://www.murphet.com",
    license="MIT",
    python_requires=">=3.8",  # ← same floor as pyproject
    keywords="bayesian time-series stan prophet churn",

    # ---- src‑layout --------------------------------------------------------
    package_dir={"": "src"},               # root of importable pkgs
    packages=find_packages("src"),         # auto‑discover under ./src
    include_package_data=True,
    package_data={"murphet": ["*.stan"]},  # ship Stan files inside wheel

    install_requires=[
        "cmdstanpy>=1.1.0",
        "numpy>=1.22",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
