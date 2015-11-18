from setuptools import setup
import versioneer

setup(
	name="recordlinkage",
	version=versioneer.get_version(),
	cmdclass=versioneer.get_cmdclass(),
  	author="Jonathan de Bruin",
	platforms="any",
	description="A tool written in Python to link and/or deduplicate small or medium sized record files.",
	url="https://github.com/J535D165/recordlinkage",
	install_requires=["pandas", "numpy", "jellyfish"],
	packages=["recordlinkage"]
	)