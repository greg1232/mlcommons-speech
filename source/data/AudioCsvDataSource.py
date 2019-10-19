
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

class AudioCsvDataSource:
    def __init__(self, config, source_config):
        self.config = config
        self.source_config = source_config

    def get_tf_dataset(self):
        line_dataset = tf.data.experimental.CsvDataset(
            self.get_path(), [tf.string, tf.string])

        text_dataset = line_dataset.map(lambda x, y :
            self.load_and_tokenize((x, y)))

        return text_dataset

    def load_and_tokenize(self, row):
        wav_file = tf.io.read_file(row[0])

        audio_samples, audio_sample_rate = tf.audio.decode_wav(wav_file)

        audio_sample_count = tf.shape(audio_samples)[0]

        label = row[1]

        sample = audio_samples, audio_sample_count, audio_sample_rate, label

        return sample

    def get_path(self):
        return self.source_config['path']

    def get_cache_file(self):
        return self.source_config['cache']


