# setup.py ─ PEP 517/518 fallback (main metadata in pyproject.toml)
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

setup(
    # -----------------------------------------------------------------
    name                = "murphet",
    version             = "1.5.2",                       # ← keep in sync
    description         = ("Bayesian time‑series model with "
                           "Beta/Gaussian heads, smooth changepoints "
                           "and seasonality"),
    long_description    = (ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type = "text/markdown",
    author              = "Stephen Murphy",
    author_email        = "stephenjmurph@gmail.com",
    license             = "MIT",
    python_requires     = ">=3.8",
    # marketing / SEO
    keywords            = ("prophet, time‑series, bayesian, stan, "
                           "beta regression, churn, forecasting"),
    project_urls        = {
        "Homepage"   : "https://murphet.com",
        "Repository" : "https://github.com/halsted312/murphet",
        "Issue Tracker": "https://github.com/halsted312/murphet/issues",
        "Documentation": "https://murphet.com/docs",
    },

    # -----------------------------------------------------------------
    # src‑layout
    package_dir         = {"": "src"},
    packages            = find_packages(where="src"),
    include_package_data= True,
    package_data        = {"murphet": ["*.stan"]},       # ship Stan models

    # hard requirements (same as [project] dependencies)
    install_requires    = [
        "cmdstanpy>=1.1.0",
        "numpy>=1.22",
        "pandas>=1.5",
        "scipy>=1.9",
    ],

    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Framework :: Jupyter",
    ],
)
