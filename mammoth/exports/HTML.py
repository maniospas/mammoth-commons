from kfp import dsl
import base64


def _encode_image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


class HTML:
    integration = dsl.HTML

    def __init__(self, body, header="", script="", images=None):
        self.body = body
        self.header = header
        self.script = script
        self.images = dict() if images is None else images

    def text(self):
        body = self.body
        for image, path in self.images.items():
            data = _encode_image_to_base64(path)
            img = f'<img src="base64,{data}" alt="{image}" />'
            body.replace(image, img)
        return body

    def export(self, output: dsl.Output[integration]):
        with open(output.path, "w") as f:
            output.name = "result.html"
            f.write(self.text())
