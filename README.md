# AITW_microstructures

Port of the AITW microstructures code from Matlab to Python.

## Install

```bash
cd <PATH to folder with pyproject.toml>
pip install .
```

## Usage

### Script

Example for birch microstructure generation.

```python
from wood_microstructure import BirchMicrostructure, BirchParams

params = BirchParams.from_json('FILE_NAME.json')

for param in params:
    birch = BirchMicrostructure(param)
    birch.generate()
```

### CLI

The installation will make available a `wood_ms` command line interface.

- Run `wood_ms --help` to see the available wood microstructures.
- Run `wood_ms WOOD_TYPE --help` flag to see all available options.

Example for birch microstructure generation:

```bash
wood_ms birch --json_file examples/example_birch.json
```

#### Tab autocompletion

Enabling tab autocompletion https://click.palletsprojects.com/en/stable/shell-completion/

E.G for `bash` run the command

```bash
eval "$(_WOOD_MS_COMPLETE=bash_source wood_ms)"
```

You can also add it to either `~/.bashrc` or, if you are using a virtual environment, to `bin/activate` of the virtual environment to avoid running the command for every new shell.

## NOTE

- W.R.T the matlab code, `saveSlice` in the json file can be a `list[int]` instead of just an `int`.
- Most functions have been optimized to have the loop over the Z slices as the outermost one.
  Since the Z slices are almost always treated independently, we can than run the loops over only the slices of interest.
- Need more testing with vessels
