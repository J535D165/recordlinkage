import os

from setuptools import setup
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
        "numpy>=1.9.0",
        "pandas>=0.18.0",
        "scipy>=0.17.1",
        "scikit-learn>=0.17.1",
    ],
    packages=[
        'recordlinkage',
        'recordlinkage.datasets',
        'recordlinkage.standardise',
        'recordlinkage.algorithms'
    ],
    include_package_data=True,
    package_dir={'recordlinkage': 'recordlinkage'},
    package_data={'recordlinkage': ['datasets/*/*.csv']},
    license='GPL-3.0'
)
