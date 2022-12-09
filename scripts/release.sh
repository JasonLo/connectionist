rm -r dist
python3 -m build
# Release to testpypi
python3 -m twine upload --repository testpypi dist/*

# Release to pypi
# python3 -m twine upload dist/*

# account: __token__
# password: get token from pypi.org