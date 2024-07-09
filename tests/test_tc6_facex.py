from mammoth import testing
from catalogue.dataset_loaders.images import data_images
from catalogue.model_loaders.pytorch import model_torch
from catalogue.metrics.xai_analysis import facex


def test_facex():
    with testing.Env(data_images, model_torch, facex) as env:

        target = "task"
        protected = "protected"
        model_path = "./data/torch_model/torch_model.py"
        model_dict = "./data/torch_model/resnet18.pt"
        data_dir = "./data/xai_images/race_per_7000"
        csv_dir = "./data/xai_images/bupt_anno.csv"

        # additional arguements needed for faceX
        target_class = 1
        target_layer = "layer4"

        dataset = env.data_images(
            path=csv_dir,
            root_dir=data_dir,
            target=target,
            data_transform="./data/xai_images/torch_transform.py",
            batch_size=1,
            shuffle=False,
        )

        model = env.model_torch(
            model_path,
            model_dict,
        )

        markdown_result = env.facex(dataset, model.model, [protected], target_class, target_layer)
        markdown_result.show()

test_facex()
