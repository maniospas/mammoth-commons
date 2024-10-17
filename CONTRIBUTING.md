# Create Modules

This document contains instructions on how to contribute modules to the MAMMOth catalogue 
so that they are included in the namesake fairness toolkit. Modules depend on the
MAMMOTH-commons library to work with its various file types. To contribute to the main
library (for example, to add data types) see [here](../mammoth-commons/README.md).

**The catalogue may be hosted in a different repository in the future.**

1. [Set things up](#set-things-up)
2. [Write a new module](#write-a-new-module)
3. [Locally test a module](#locally-test-a-module)
4. [Build and upload a module](#build-and-upload-a-module)

## Set things up

**Installation:** Install the latest version of `MAMMOth-commons`
and the `docker` package in your virtual environment:

*If you are working in your own repository:*

```bash
pip install --upgrade MAMMOth-commons
pip install docker
```

*If you are working in a clone of mammoth-commons and plan to create a pull request:*

```bash
pip install -e .
pip install requirements[test].txt # needed only if you want to run all integration tests
pip install docker
```

**New account:** You also need to create an account in
[DockerHub](!https://hub.docker.com/) or any other online
hosting service for docker images. You can ignore this step
while developing or testing modules.

**Required tools:** Finally, download, install, and run Decker Desktop
from [here](https://docs.docker.com/get-docker/). Command 
line instructions will use this to build docker images locally
before uploading them to the hosting service. You can also skip
this at the first stages of development.

## Write a new module

You need to have set everything up as above to build and
deploy your MAMMOth modules. Follow the next steps
to write a module by implementing a base function, adding typehints
and documentation, and wrapping it with a decorator at the end.
The decorator will automate all preparation needed to convert your
method to a proper module.

1. *Type dependencies.* Import the necessary dataset or model classes
from the `mammoth.datasets` and `mammoth.models` namespace respectively. 
Use them to annotate your method's argument
and return types. *Type annotations are mandatory for 
all arguments.*

2. *Parameters.* In addition to some mandatory positional
arguments for each type of module, you may add any number of 
`str`, `bool`, `int` or `float` keyword arguments. These
serve as parameters with default values, where the default `None` should
be set if no common default is known beforehand. You must also create
a docstring for your module, which should include both its main description
and parameter descriptions under an `Args:`
section. The parameter descriptions should follow the convention `name: description` and not
specify any type. In case a string is an enumeration, you can replace the `str` type with
`Options("option1", "option2", ...)`. All these requirements help the toolkit understand
what information to give to the users working with your module.

3. *Decorators.* Decorate your module with either the 
`@mammoth.integration.metric(namespace, version, python="3.11", packages=(...))` or 
the `@mammoth.integration.loader(namespace, version, python="3.11", packages=(...))` decorator. 
These require at least one argument to denote
the module's version. The namespace refers to whom the module
should be accredited to and should be the same as your DockerHub 
username. Finally, packages are library dependencies and must be a tuple of strings 
(take care to write something like `packages=("pandas",)` **comma included** if you only have one dependency).
These dependencies are any packages other than the few found in `requirements.txt`.
Notice that mammoth-commons imports packages for its datatypes only at the
last necessary moment.

Here are some examples of modules:

<details>
<summary>Example metric</summary>

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
    return Markdown("#Results\nThese are the results.")

```
</details>


<details>
<summary>Example dataset loader</summary>

```python
from mammoth.datasets import CSV
from mammoth.integration import loader
import fairbench as fb
from typing import List, Optional


@loader(
    namespace="maniospas",
    version="v001",
    python="3.11",
    packages=("pandas",),
)
def categorical_csv(
    path: str = "",
    categorical: Optional[List[str]] = None, 
    label: Optional[str] = None,
) -> CSV:
    """Loads a CSV file that contains categorical and predictive data columns.

    Args:
        path: The local file path or a web URL of the file.
        categorical: A list of column names that hold categorical data.
        label: The name of the categorical column that holds predictive label for each data sample.
    """
    import pandas as pd  # safe import here
    ...
    return CSV(...)
```
</details>


<details>
<summary>Example model loader</summary>

```python
from mammoth.models import ONNX
from mammoth.integration import loader

@loader(namespace="...", version="v001", python="3.11")
def model_onnx(
    path: str
) -> ONNX:
    """This is an ONNX loader.
    """
    return ONNX(path)

```
</details>

## Locally test a module

After decorating a module, you will want to test that it
runs correctly before uploading it for public consumption.
To write tests that
verify but then ignore your decorators to run on local data, 
create a context from which you can access the undecorated methods 
like so:

```Python
import mammoth
from modules import dataloader, modelloader, metric  # import your modules here

with mammoth.testing.Env(dataloader, modelloader, metric) as env:
    data = env.dataloader("data_url", data_kwarg1=..., data_kwarg2=..., ...)
    model = env.dataloader("model_url", model_kwarg1=..., model_kwarg2=..., ...)
    sensitive = ["attr1", "attr2", ...]  # list of sensitive attributes
    result = env.metric(data, model, sensitive, metric_kwarg1=..., metric_kwarg2=..., ...) 
    print(result.text)
```

If you are planning to create a pull request to mammoth-commons, also
create a file `tests/test_...` containing the above code. This will be run
by the `integration_tests.py` script. Everything new is expected to have high 
code coverage (more than 80% right now).

Do not forget to add all requirements to the `requirements[test].txt` file.
Pull requests will be reviewed manually, so if you plan to create a complex one
get in touch with us by opening an issue first. Finally, your module should 
be automatically added to the demonstrator do that you can see how it is going
to appear in the main toolkit 
(instructions on launching the demonstrator [here](../mammoth-commons/README.md)).
The demonstrator does not require the steps covered below and you can run it
during development.

## Build and upload a module

Don't forget to set the correct module version first (if you reuse 
a previously uploaded version, the toolkit may not be able to see the change).
Then, [login to your docker account](https://docs.docker.com/engine/reference/commandline/login/).
For example, in the simplest case where you want to host your module
in DockerHub, it suffices to run the following command in your terminal:

```bash
docker login
```

This will ask for your DockerHub username (if you are not part of
a team in DockerHub, this should be the same as your namespace) 
and password. This way, your terminal will have
permission to push the created docker images there. 

Also make
sure that the library is visible to your virtual environment by calling
in the top level (from where you can access subdirecories 
*mammoth/*, *catalogue/*, *tests/*, etc)

```bash
pip install -e .
```


Finally, create and upload a module by running the following
command (kfp is installed alongside MAMMOth-commons):

```bash
kfp module build . --module-filepattern catalogue/fairbench/modelcard.py 
```

In this, replace the `test_modules/metric.py` with any other path
that contains the Python file in which you implemented your module. 

If you do *not* want to push the created docker image, for
example to run your new module in a local copy of the MAMMOth
bias toolkit without logging in and uploading it to DockerHub, run
this instead:

```bash
kfp module build . --module-filepattern catalogue/fairbench/modelcard.py --no-push-image
````

:warning: The build should be called from a directory where both your
module and virtual environment are subdirectories.
