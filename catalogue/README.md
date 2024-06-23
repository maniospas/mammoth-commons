# Component Catalogue

This directory hosts a number of components that are
supported by the MAMMOth-commons library, and which
are subsequently integrated in the MAMMOth toolkit's 
catalogue.
Either create pull requests to add more components here, 
or create new repositories for those components.

:warning: If you need new data types to represent
the outcomes of loaders, these datatypes should be integrated
within the core of the MAMMOth-commons package. This way,
they can be imported from the `mammoth` module once
the package is installed within docker containers, which
in turn facilitates communication between loaders and 
metrics by referencing the same data types.

There are three subdirectories in which you can find
and place components of respective types:

- `dataset_loaders/` contains components for loading datasets in various formats and internally converting them to some standard representations.
- `model_loaders/` contains components for loading various types of machine learning or other AI models.
- `metrics/` contains components that take as inputs a dataset loader and a model loader, perform some type of analysis using those, and output HTML or markdown.

The following components are currently presented in the catalogue:

| Component                        | Input datatypes        | Output datypes | Parameters                                                                              |
|----------------------------------|------------------------|----------------|-----------------------------------------------------------------------------------------|
| `dataset_loaders/autocsv.py`     |                        | CSV            | Path and options to provide to Pandas.                                                  |
| `dataset_loaders/autocsv.py`     |                        | Image          | Path for csv attributes and image location.                                             |
| `model_loaders/rankings.py`      |                        | 
| `model_loaders/onnx.py`          |                        | ONNX           | Path to the stored model.                                                               |
| `model_loaders/onnx_ensemble.py` |                        | ONNXEnsemble   | Path to the stored model.                                                               |
| `model_loaders/pytorch.py`       |                        | Pytorch        | Path or code to the torch model's construction, and path to the torch state dictionary. |
| `metrics/image_bias_analysis.py` | Image, Pytorch         | Markdown       |                                                                                         |
| `metrics/model_card.py`          | Any dataset, any model | Markdown       |                                                                                         |
| `metrics/xai_analysis.py`        | Image, Pytorch         | HTML           |                                                                                         |
