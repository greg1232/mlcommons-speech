
import os

class PredictorFactory:
    def __init__(self, config, test_data):

        self.config = config
        self.test_data = test_data

    def create(self):

        raise RuntimeError("Unknown predictor type ")

