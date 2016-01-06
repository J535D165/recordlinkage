nosetests

cd docs/
ipython nbconvert --to rst ../examples/*.ipynb

make html