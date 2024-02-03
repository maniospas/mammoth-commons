from kfp import dsl
import inspect
from typing import get_type_hints
from typing import Dict, List
import pickle
import yaml
import os
from collections import OrderedDict


def metric(version, python="3.11", packages=()):
    def wrapper(method):
        # prepare the kfp wrapper given decorator arguments
        name = method.__name__  # will use this as the component id
        base_image = f"python:{python}-slim-bullseye"
        target_image = f"mammotheu/metric_{name}: {version}"
        kfp_wrapper = dsl.component(
            base_image=base_image,
            target_image=target_image,
            packages_to_install=["mammoth-commons"]+list(packages),
        )

        # find signature and check that we can obtain the integration type from the returned type
        signature = inspect.signature(method)
        return_type = signature.return_annotation
        if return_type is inspect.Signature.empty:
            raise Exception(f"The metric {name} must declare a return type")
        if not hasattr(return_type, "integration"):
            raise Exception(f"Missing static field in the return type of {name}: {return_type.__name__}.integration")

        # keep type hint names, keeping default kwargs (these will be kwarg parameters)
        type_hints = get_type_hints(method)
        defaults = dict()
        input_types = list()
        for pname, parameter in signature.parameters.items():
            if pname == "sensitive":  # do not consider the sensitive attributes for component types
                continue
            arg_type = type_hints.get(pname, parameter.annotation)
            if parameter.default is not inspect.Parameter.empty:  # ignore kwargs
                defaults[pname] = parameter.default
                continue
            if pname not in ["dataset", "model"]:
                raise Exception("Only `dataset`, `model`, `sensitive` and keyword arguments are supported for metrics")
            if arg_type is inspect.Signature.empty:
                raise Exception(f"Add a type annotation in method {name} for the argument: {pname}")
            input_types.append(arg_type.__name__)
            #print(f"Argument: {pname}, Type: {arg_type.__name__}")
        if len(input_types) != 2:
            raise Exception("Your metric should have both a `dataset` and `model` arguments")

        # create component_metadata/{name}_meta.yaml
        metadata = {
            "id": name,
            "name": " ".join(name.split("_")),
            "description": method.__doc__,
            "parameter_info": "No parameters needed." if not defaults else "Some parameters are needed.",
            "component_type": "METRIC",
            "input_types": input_types,
            "output_types": []  # no kfp output, the data are exported when running the metric
        }
        if not os.path.exists("component_metadata/"):
            os.makedirs("component_metadata/")
        with open(f'component_metadata/{name}_meta.yaml', 'w') as file:
            yaml.dump(metadata, file, sort_keys=False)


        # create the kfp method to be wrapped
        def kfp_method(dataset: dsl.Input[dsl.Dataset],
                       model: dsl.Input[dsl.Model],
                       output: return_type.integration,
                       sensitive: List[str],
                       parameters: Dict[str, any] = defaults,
                       ):
            with open(dataset.path, "rb") as f:
                dataset_instance = pickle.load(f)
            with open(model.path, "rb") as f:
                model_instance = pickle.load(f)
            parameters = {**defaults, **parameters}  # insert missing defaults into parameters (TODO: maybe this is not needed)
            ret = method(dataset_instance, model_instance, output.path, sensitive, **parameters)
            assert isinstance(ret, return_type)
            ret.export(output)

        # rename the kfp_method so that kfp will create an appropriate name for it
        kfp_method.__name__ = name

        # return the wrapped kfp method
        return kfp_wrapper(kfp_method)
    return wrapper

