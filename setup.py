from setuptools import setup, find_packages
import versioneer

setup(
	name="recordlinkage",
	version=versioneer.get_version(),
	cmdclass=versioneer.get_cmdclass(),
  	author="Jonathan de Bruin",
  	author_email="jonathandebruinhome@gmail.com",
	platforms="any",
	description="A tool to link or deduplicate small or medium sized datasets.",
	url="https://github.com/J535D165/recordlinkage",
	install_requires=["numpy", "pandas", "scipy", "sklearn"],
	packages=['recordlinkage', 'recordlinkage.datasets', 'recordlinkage.standardise'],
	include_package_data=True,
	package_dir={'recordlinkage': 'recordlinkage'},
	package_data={'recordlinkage': ['datasets/data/*.csv']},
)