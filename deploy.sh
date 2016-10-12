#!/bin/bash
set -e

if ! git diff-index --quiet HEAD --; then

    echo "There are uncommited changes, do you want to continue (y/n)? "
    read cont
    if echo "$cont" | grep -iq "^n" ;then
        exit
    fi
fi

echo "The current versions are: "
git tag

echo ""
echo "Give a new version number (without v prefix)"
read version_tag

echo ""
git tag -a "v$version_tag" -m "Version $version_tag"

# Make the package installable. 
python setup.py bdist_wheel

# recordlinkage-0.6.0+0.g9c83c85.dirty-py2.py3-none-any.whl
base_path="dist/recordlinkage-"
ext_path="-py2.py3-none-any.whl"
full_path=$base_path$version_tag$ext_path
echo $full_path

echo ""
echo "Upload release to PiPy (y/n)? "
read upload_pip

if echo "$upload_pip" | grep -iq "^y" ;then

    twine upload $full_path

fi