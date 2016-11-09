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

    # Documentation
    url="http://recordlinkage.readthedocs.io/",

    install_requires=["numpy", "pandas", "scipy", "sklearn"],
    packages=[
        'recordlinkage',
        'recordlinkage.datasets',
        'recordlinkage.standardise',
        'recordlinkage.algorithms',
        'recordlinkage.compat'
    ],
    include_package_data=True,
    package_dir={'recordlinkage': 'recordlinkage'},
    package_data={'recordlinkage': ['datasets/*/*.csv']},
    license='GPL-3.0'
)
