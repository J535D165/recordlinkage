# Run in terminal with: sh tests.sh

nosetests

cd docs/
ipython nbconvert --to rst ../examples/*.ipynb

make html