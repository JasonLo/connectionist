#!/bin/bash

if [ "$1" = "" ]; then
    echo "Usage: release.sh major/minor/patch... (see hatch version --help for details)"
    exit 1
fi

# Bump version with hatchling
hatch version $1

# Get version number
RELEASE_VERSION=`hatch version`

rm -r dist
python3 -m build

# Release to pypi
python3 -m twine upload dist/*

# Commit version bump
git add connectionist/__about__.py
git commit -m "bump version to $RELEASE_VERSION"

# Tag in Git
git tag $RELEASE_VERSION -m "release $RELEASE_VERSION"
git push --tags

# Remember to add the following to ~/.pypirc
# [pypi]
# username = __token__
# password = <token from pypi.org>