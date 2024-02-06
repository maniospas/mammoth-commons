from kfp import dsl


class Markdown:
    integration = dsl.Markdown

    def __init__(self, text):
        self.text = text

    def export(self, output: dsl.Output[integration]):
        with open(output.path, "w") as f:
            output.name = "result.md"
            f.write(self.text)
