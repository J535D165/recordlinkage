import os

from setuptools import setup, find_packages
import versioneer


def read(fname):
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
    ],
    packages=find_packages(exclude=["benchmarks", "tests", "docs"]),
    include_package_data=True,
    package_data={'recordlinkage': ['datasets/*/*.csv']},
    license='BSD-3-Clause'
)
