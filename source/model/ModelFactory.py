

import os

from model.TransformerModel import TransformerModel

class ModelFactory:
    def __init__(self, config,
        training_data, validation_data):

        self.config = config
        self.model_name = config["model"]["type"]
        self.validation_data = validation_data
        self.training_data = training_data

    def create(self):

        if self.model_name == "TransformerModel":
            return TransformerModel(self.config,
                self.training_data, self.validation_data)

        raise RuntimeError("Unknown model name " + self.model_name)




