# MAMMOth-commons

[![Integration Tests](https://github.com/mammoth-eu/mammoth-commons/actions/workflows/integration.yml/badge.svg)](https://github.com/mammoth-eu/mammoth-commons/actions/workflows/integration.yml)
![Coverage](./coverage-badge.svg)

Fast component development for the MAMMOth fairness toolkit.
Components refer to model loaders, dataset loaders, or metrics.
The library holds common datatypes that are shared between
components, and automates the integration strategy by only
needing to add a decorator.

*Some demonstration data are large and stored with LFS.
To clone this repo without retrieve those files and only
later retrieve them use:*

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mammoth-eu/mammoth-commons.git
git lfs pull --include "./data/torch_model/ir50_adaface.pth"  # get a large file from lfs
```


## :microscope: Investigate fairness

This repo includes a thinned down variation of the MAMMOth toolkit
that you can quickly install and run locally. Instructions to launch
the web application locally in your machine:

1. Download this repository.
2. Create a virtual environment. This is optional but recommended.
3. Install dependencies with `pip install -r requirements.txt`. This can take a bit of time to download and install everything, but you will be able to run all modules and interface with most popular data types.
4. Launch the local app server with `python demonstrator/app.py`. When everything is ready, this script will also open a browser window to the app's serving page at `http://localhost:5050`.

## :clipboard: Catalogue

Find a catalogue of modules implemented by the MAMMOth consortium
[here](catalogue/README.md). These modules are developed by and
depend on datatypes found in commons.

## :thumbsup: Contributing

Instructions on how to add new modules are [here](CONTRIBUTING.md).
Use the GitHub issue tracker to ask questions, request 
features/improvements for the core library or modules, or report bugs.
