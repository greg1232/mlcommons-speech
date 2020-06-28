
from dataset.AudioCsvDataset import AudioCsvDataset
from dataset.TextCsvDataset  import TextCsvDataset

class DatasetFactory:
    def __init__(self, config):
        self.config = config

    def create(self, source_description):

        if source_description["type"] == "AudioCsvDataset":
            return AudioCsvDataset(self.config, source_description)

        if source_description["type"] == "TextCsvDataset":
            return TextCsvDataset(self.config, source_description)

        raise RuntimeError("Unknown data source type '" +
            source_description["type"] + "'")







