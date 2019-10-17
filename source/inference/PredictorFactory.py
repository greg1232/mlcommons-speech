
import os

class PredictorFactory:
    def __init__(self, config, validation_data):

        self.config = config
        self.validation_data = validation_data

    def create(self):

        raise RuntimeError("Unknown predictor type ")

