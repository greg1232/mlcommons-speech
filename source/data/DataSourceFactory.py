
from data.AudioCsvDataSource import AudioCsvDataSource

class DataSourceFactory:
    def __init__(self, config):
        self.config = config

    def create(self, source_description):

        if source_description["type"] == "AudioCsvDataSource":
            return AudioCsvDataSource(self.config, source_description)

        raise RuntimeError("Unknown data source type '" +
            source_description["type"] + "'")







