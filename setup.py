from setuptools import setup

setup(
	name="recordlinkage",
	version="0.0.1",
	author="Jonathan de Bruin",
	platforms="any",
	description="A tool written in Python to link and/or deduplicate small or medium sized record files.",
	url="https://github.com/J535D165/recordlinkage",
	install_requires=["pandas", "numpy", "jellyfish"],
	packages=["recordlinkage"]
	)