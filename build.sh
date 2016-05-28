# Run in terminal with: sh build.sh

virtualenv -p /usr/bin/python2.7 env/python2
virtualenv -p /usr/bin/python3.5 env/python3

source env/python2/bin/activate
pip install -r requirements.txt

pip install mock nbsphinx

nosetests

deactivate
rm -rf env/python2

source env/python3/bin/activate
pip install -r requirements.txt

nosetests

deactivate
rm -rf env/python3

# Create the documentation.
pandoc --from=markdown_github --to=rst --output=docs/README.rst README.md

cd docs/
jupyter nbconvert --to rst ../examples/*.ipynb

make clean
make html
make latexpdf

cd ..

# Make the package installable. 
python setup.py sdist
python setup.py build
python setup.py install