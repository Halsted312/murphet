import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="murphet",
    version="0.1.3",
    author="Stephen Murphy",
    author_email="stephenjmurph@gmail.com",
    description="A Bayesian time-series model for churn rates with changepoints and seasonality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/halsted312/murphet",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "murphet": ["*.stan"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="bayesian, time-series, prophet, stan, churn",
    python_requires=">=3.7",
    install_requires=[
        "cmdstanpy>=0.10.0",
        "numpy>=1.19",
        "pandas>=1.0.0"
    ],
)
