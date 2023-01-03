#!/bin/sh

# Bump version with hatchling
RELEASE_VERSION=`hatch version $1`

rm -r dist
python3 -m build

# Release to pypi
python3 -m twine upload dist/*

# Tag in Git
git tag $RELEASE_VERSION -m "Release $RELEASE_VERSION"
git push --tags

# Remember to add the following to ~/.pypirc
# [pypi]
# username = __token__
# password = <token from pypi.org>