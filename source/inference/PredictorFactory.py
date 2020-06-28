
from inference.GreedyDecoder import GreedyDecoder
from inference.BeamSearchDecoder import BeamSearchDecoder

class PredictorFactory:
    def __init__(self, config, test_dataset):

        self.config = config
        self.test_dataset = test_dataset

    def create(self):

        if self.config["predictor"]["type"] == "GreedyDecoder":
            return GreedyDecoder(self.config, self.test_dataset)
        elif self.config["predictor"]["type"] == "BeamSearchDecoder":
            return BeamSearchDecoder(self.config, self.test_dataset)

        raise RuntimeError("Unknown predictor type '" + self.config["predictor"]["type"] + "'")

