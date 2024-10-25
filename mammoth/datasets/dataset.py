class Dataset:
    integration = "dsl.Dataset"

    def to_features(self, sensitive):
        raise Exception(
            f"{self.__class__.__name__} has no method to_features(sensitive)"
        )
