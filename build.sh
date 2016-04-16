# Run in terminal with: sh build.sh

# Create the documentation.
pandoc --from=markdown_github --to=rst --output=docs/README.rst README.md

cd docs/
jupyter nbconvert --to rst ../examples/*.ipynb

make clean
make html
make latexpdf

# Make the package installable. 
cd ..
python setup.py sdist
python setup.py build
python setup.py install