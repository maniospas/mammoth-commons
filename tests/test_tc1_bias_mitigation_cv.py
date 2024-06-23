import mammoth
from catalogue.loaders.images import data_images
from catalogue.loaders.pytorch import model_torch
from catalogue.metrics.image_bias_analysis import image_bias_analysis


def test_facex():
    with mammoth.testing.Env(data_images, model_torch, image_bias_analysis) as env:

        target = "task"
        task = "image classification"  # or "face verification" TODO: error on unknown tasks
        protected = "protected"
        data_dir = "./data/xai_images/race_per_7000"
        csv_dir = "./data/xai_images/bupt_anno.csv"

        dataset = env.data_images(
            path=csv_dir,
            root_dir=data_dir,
            target=target,
            data_transform="./data/xai_images/torch_transform.py",
            batch_size=1,
            shuffle=False,
        )
        model = env.model_torch("./data/torch_model/model.py", "./data/torch_model/model.pth")
        analysis_outcome = env.image_bias_analysis(
            dataset, model, [protected], task
        )
        print(analysis_outcome.text)


test_facex()
