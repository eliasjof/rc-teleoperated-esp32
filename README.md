# Package - DE path planners 3D

## Generate the .whl file (package)

```bash
python setup.py bdist_wheel
``` 

## Install package

Observation: Check the version of the package

```bash
pip install .\dist\
```

If this package is already installed, you can use this:

```bash
pip install .\dist\--force-reinstall
```
## Requirements

To get the list of requirements of a project:

```bash
pipreqs .
```
