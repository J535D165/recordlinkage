# Run in terminal with: sh build.sh

cd docs/
jupyter nbconvert --to rst ../examples/*.ipynb

make html

cd ..

python setup.py sdist
python setup.py build
python setup.py install