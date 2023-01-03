#!/bin/sh

# Get release version
RELEASE_VERSION=`grep version pyproject.toml | cut -d '"' -f 2`

rm -r dist
python3 -m build

# Release to pypi
python3 -m twine upload dist/*

# Tag in Git version
git tag $RELEASE_VERSION -m "Release $RELEASE_VERSION"
git push --tags

# Remember to add the following to ~/.pypirc
# [pypi]
# username = __token__
# password = <token from pypi.org>