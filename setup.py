"""Setup file for the Python Record Linkage Toolkit."""

import os
import sys

from setuptools import find_packages, setup

import versioneer


def read(fname):
    """Read a file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_min_pandas_version():
    """Get the minimum required pandas version"""

    if sys.version_info[0] < 3:
        raise Exception("This module is Python 3 only.")

    if sys.version_info[1] >= 8:
        # https://github.com/pandas-dev/pandas/issues/27261
        return "0.25"

    return "0.23"


setup(
    name="recordlinkage",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Jonathan de Bruin",
    author_email="jonathandebruinhome@gmail.com",

    platforms="any",

    # Description
    description="A record linkage toolkit for linking and deduplication",
    long_description=read('README.rst'),

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
    python_requires=">=3.5",
    install_requires=[
        "jellyfish>=0.5.4",
        "numpy>=1.13.0",
        "pandas>=" + get_min_pandas_version(),
        "scipy>=1",
        "scikit-learn>=0.19.0",
        "joblib"
    ],
    packages=find_packages(
        exclude=["benchmarks", "docs",
                 "*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    include_package_data=True,
    package_data={'recordlinkage': ['datasets/*/*.csv']},
    license='BSD-3-Clause'
)
