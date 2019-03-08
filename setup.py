"""Setup file for the Python Record Linkage Toolkit."""

import os

from setuptools import find_packages, setup

import versioneer


def read(fname):
    """Read a file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


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

    install_requires=[
        "six>=1.10.0",
        "jellyfish>=0.5.4",
        "numpy>=1.13.0",
        "pandas>=0.18.0",
        "scipy>=0.17.1",
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
