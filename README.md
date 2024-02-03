# MAMMOth-commons

Component interfaces of the MAMMOth fairness toolkit.

**This package is in the pre-alpha stage.**

A first version will be released with the first 
version of the toolkit.

## How to create a new component

Install the latest version of `MAMMOth-commons`
in your virtual environment:

```bash
pip install --upgrade MAMMOth-commons
```

Import the necessary dataset or models from the `mammoth`
namespace and use them to annotate your method's inputs
and outputs, like in the snippet bellow. 
*Annotations are mandatory for these data types.* 

You may have additional keyword arguments without annotation.
Don't forget to create a docstring for your component too.

In the end, decorate your component with our `metric` decorator,
proving a version.

```python
from mammoth.datasets import CSV
from mammoth.models import ONNX
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric


@metric(version="v001")
def new_metric(
    dataset: CSV,
    model: ONNX,
    sensitive: List[str],
    parameters: Dict[str, any] = None,
) -> Markdown:
    """
    Write your metric's description here.
    """
    return Markdown("these are the results")

```

You can then create a technical component by running the following
command (to run this, also run `pip install docker` first):
```bash
kfp component build . --component-filepattern test.py --no-push-image
```