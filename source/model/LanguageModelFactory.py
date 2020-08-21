

from model.TransformerLanguageModel import TransformerLanguageModel

class LanguageModelFactory:
    def __init__(self, config,
        training_data=None, validation_data=None):

        self.config = config
        self.model_name = config["language-model"]["type"]
        self.validation_data = validation_data
        self.training_data = training_data

    def create(self):

        if self.model_name == "TransformerLanguageModel":
            return TransformerLanguageModel(self.config,
                self.training_data, self.validation_data)

        raise RuntimeError("Unknown model name " + self.model_name)





