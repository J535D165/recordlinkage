# Run in terminal with: sh build.sh

# Create the documentation.
cd docs/
jupyter nbconvert --to rst ../examples/*.ipynb

make html
make latexpdf

# Make the package installable. 
cd ..
python setup.py sdist
python setup.py build
python setup.py install