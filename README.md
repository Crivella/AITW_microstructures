# AITW_microstructures

Port of the AITW microstructures code from Matlab to Python.

## Install

```bash
cd <PATH to folder with pyproject.toml>
pip install .
```

## Usage

### Script

```python
from birch_data_generation import BirchMicrostructure, RayCellParams

params = RayCellParams.from_json('FILE_NAME.json')

for param in params:
    birch = BirchMicrostructure(param)
    birch.generate()
```

### CLI

The installation will make available a `woodms` command line interface.

Use the `woodms birch --help` flag to see all available options.

```bash
woodms birch -v --json_file INPUT_FILE_PATH.json --max-parallel 2
```

#### Tab autocompletion

Enabling tab autocompletion https://click.palletsprojects.com/en/stable/shell-completion/

E.G for `bash` add the following

```bash
eval "$(_WOOD_MS_COMPLETE=bash_source wood_ms)"
```

to either `~/.bashrc` or if you are using a virtual environment to `bin/activate` of the virtual environment.

## NOTE

- W.R.T the matlab code, `saveSlice` in the json file can be a `list[int]` instead of just an `int` and the values are **0-indexed** instead of 1-indexed.
- Most functions have been optimized to have the loop over the Z slices as the outermost one.
  Since the Z slices are almost always treated independently, we can than run the loops over only the slices of interest.
- Need more testing with vessels
