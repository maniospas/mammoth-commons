# MAMMOth-commons

[![Integration Tests](https://github.com/mammoth-eu/mammoth-commons/actions/workflows/integration.yml/badge.svg)](https://github.com/mammoth-eu/mammoth-commons/actions/workflows/integration.yml)
![Coverage](./coverage-badge.svg)

Fast module development for the MAMMOth fairness toolkit.
Modules refer to model loaders, dataset loaders, or metrics.
The library holds common datatypes that are shared between
modules, and automates the integration strategy by only
needing to add a decorator. It also provides integration
tests, as well as a lightweight demonstrator that is a thinned
down version of the toolkit.


<details>
<summary>
<i>Some demonstration data are large and stored with LFS.
Here is how to clone this repo without retrieving 
those files. You can pull them on demand.</i>
</summary>

*Linux*
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mammoth-eu/mammoth-commons.git
git lfs pull --include "./data/torch_model/ir50_adaface.pth"  # get a large file from lfs
```

*Windows*
```bash
set set GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/mammoth-eu/mammoth-commons.git
git lfs pull --include "./data/torch_model/ir50_adaface.pth"  # get a large file from lfs
```

</details>



## :microscope: Investigate fairness

Instructions to quickly launch and install the demonstrator 
web application locally in your machine:

1. Download this repository.
2. Create a virtual environment. This is optional but recommended.
3. Install dependencies with `pip install -r requirements[test].txt`. This can take a bit of time to download and install everything, but you will be able to run all modules and interface with most popular data types.
4. Launch the local app server with `python demonstrator/app.py`. When everything is ready, this script will also open a browser window to the app's serving page at `http://localhost:5050`.

## :clipboard: Catalogue

Find a catalogue of modules implemented by the MAMMOth consortium
[here](catalogue/README.md). These modules are developed by and
depend on datatypes found in the main commons library that resides 
under this repo's `mammoth/` directory.

## :thumbsup: Contributing

Instructions on how to add new modules are [here](CONTRIBUTING.md).
Use the GitHub issue tracker to ask questions, request 
features/improvements for the core library or modules, or report bugs.
