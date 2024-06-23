import mammoth
from catalogue.loaders.images import data_images
from catalogue.loaders.pytorch import model_torch
from catalogue.xai_analysis import facex


def test_facex():
    with mammoth.testing.Env(data_images, model_torch, facex) as env:

        target = "task"
        protected = "protected"
        model_path = "./catalogue/model"
        model_dict = "resnet18.pt"
        data_dir = "./data/xai_images/race_per_7000"
        csv_dir = "./data/xai_images/bupt_anno.csv"

        # additional arguements needed for faceX
        target_class = 1
        target_layer = "layer4"

        dataset = env.data_images(
            path=csv_dir,
            root_dir=data_dir,
            target=target,
            data_transform="",
            batch_size=1,
            shuffle=False,
        )

        model = env.model_torch(
            model_path,
            model_dict,
        )

        env.facex(dataset, model.model, [protected], target_class, target_layer)


test_facex()
