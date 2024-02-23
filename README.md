# MAMMOth-commons

Fast component development for the MAMMOth fairness toolkit.

**This package is in the pre-alpha stage.**

This file contains instructions on how to:
1. [Set things up](#set-things-up)
2. [Write a new component](#write-a-new-component)
3. [Build and upload a component](#build-and-upload-a-component)

## Set things up

Install the latest version of `MAMMOth-commons`
and the `docker` package in your virtual environment:

```bash
pip install --upgrade MAMMOth-commons
pip install docker
```

You also need to create an account in
[DockerHub](!https://hub.docker.com/) or any other online
hosting service for docker images. You can ignore this

Finally, download, install, and run Decker Desktop
from [here](https://docs.docker.com/get-docker/). Command 
line instructions will use this to build docker images locally
before uploading them to the hosting service.

## Write a new component

You need to have set everything up as above to build and
deploy your MAMMOth components. Follow these guidelines
to write a component:

1. Import the necessary dataset or model classes
from the `mammoth.datasets`
and `mammoth.models` namespace respectively. 
Use them to annotate your method's argument
and return types. *Type annotations are mandatory for 
datasets and models.*
3. You may also add keyword arguments that serve
as parameters with default values, which don't require annotation.
4. Don't forget to create a docstring for your component.
5. Decorate your component with either the 
`@mammoth.integration.metric(namespace, version, python="3.11")` or 
the `@mammoth.integration.loader(namespace, version, python="3.11")` decorator. 
These decorators require at least one argument to denote
the component's version. The namespace refers to whom the component
should be accredited to and should be the same as your DockerHub 
username.

Here are some examples of components:

<details>
<summary>Example metric decorator</summary>

```python
from mammoth.datasets import CSV
from mammoth.models import ONNX
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric


@metric(namespace="...", version="v001", python="3.11")
def new_metric(
    dataset: CSV,
    model: ONNX,
    sensitive: List[str],
    parameters: Dict[str, any] = None,
) -> Markdown:
    """Write your metric's description here.
    """
    return Markdown("these are the results")

```
</details>


<details>
<summary>Example loader decorator</summary>

```python
from mammoth.datasets import CSV
from mammoth.integration import loader

@loader(namespace="...", version="v001", python="3.11")
def data_csv_loader(
    path: str,
    delimiter: str = ",",
) -> CSV:
    """This is a CSV loader.
    """
    # load from path given delimiter or other arguments
    return CSV(
        ...  # add arguments here
    )
```
</details>

## Build and upload a component

Don't forget to set the correct component version first.
Then, [login to your docker account](https://docs.docker.com/engine/reference/commandline/login/).
For example, in the simplest case where you want to host your component
in DockerHub, it suffices to run the following command in your terminal:

```bash
docker login
```

This will ask for your DockerHub username (if you are not part of
a team in DockerHub, this should be the same as your namespace) 
and password. This way, your terminal will have
permission to push the created docker images there. Finally,
create and upload a technical component by running the following
command (kfp is installed alongside MAMMOth-commons):

```bash
kfp component build . --component-filepattern test_components/metric.py 
```

In this, replace the `test_components/metric.py` with any other path
that contains the Python file in which you implemented your component. 

If you do *not* want to push the created docker image, for
example to run your new component in a local copy of the MAMMOth
bias toolkit without logging in and uploading it to DockerHub, run
this instead:

```bash
kfp component build . --component-filepattern test_components/metric.py --no-push-image
````

:warning: The build should be called from a directory where both your
component and virtual environment are subdirectories.