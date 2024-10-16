from mammoth import testing
from catalogue.model_loaders.onnx_ensemble import model_onnx_ensemble


def test_multiattribute_bias_mitigation():
    with testing.Env(model_onnx_ensemble) as env:
        model_path = "./data/mfppb.zip"
        model = env.model_onnx_ensemble(model_path)
        print(model)


if __name__ == "__main__":
    test_multiattribute_bias_mitigation()
