"""Setup file for the Python Record Linkage Toolkit."""

from pathlib import Path

from setuptools import find_packages, setup

import versioneer


# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


# Optional dependencies for the recordlinkage package
OPTIONAL_DEPS = [
    "networkx>=2",  # clustering and hard matching
    "bottleneck",  # performance
    "numexpr"  # performance
]

setup(
    name="recordlinkage",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Jonathan de Bruin",
    author_email="jonathandebruinhome@gmail.com",

    platforms="any",

    # Description
    description="A record linkage toolkit for linking and deduplication",
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Github
    url="https://github.com/J535D165/recordlinkage",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only"
    ],

    # Python version in line with pandas' python version support
    # https://pandas.pydata.org/docs/getting_started/install.html
    python_requires=">=3.6",
    install_requires=[
        "jellyfish>=0.8.0",
        "numpy>=1.13.0",
        "pandas>=1,<2",
        "scipy>=1",
        "scikit-learn>=0.19.0",
        "joblib"
    ],
    extras_require={
        "all": OPTIONAL_DEPS,
        "test": ["pytest"] + OPTIONAL_DEPS
    },
    packages=find_packages(
        exclude=["benchmarks", "docs",
                 "*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    include_package_data=True,
    package_data={'recordlinkage': ['datasets/*/*.csv']},
    license='BSD-3-Clause'
)
