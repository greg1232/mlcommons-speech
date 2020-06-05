
import tensorflow as tf
import os
import math

AUTOTUNE = tf.data.experimental.AUTOTUNE

import logging

logger = logging.getLogger(__name__)

class AudioCsvDataset:
    def __init__(self, config, source_config):
        self.config = config
        self.source_config = source_config

    def get_tensorflow_dataset(self):
        line_dataset = tf.data.experimental.CsvDataset(
            self.get_path(), [tf.string, tf.string])

        if self.has_maximum_size():
            line_dataset = line_dataset.take(self.get_maximum_size())

        dataset = line_dataset.map(lambda x, y :
            self.load_file(x, y))

        dataset = dataset.cache(filename=self.get_cache_path())

        dataset = dataset.shuffle(self.get_shuffle_window_size(), seed=42)
        #dataset = self.group_by_sequence_length(dataset)

        dataset = dataset.padded_batch(self.get_mini_batch_size(), drop_remainder=True)
        dataset = dataset.prefetch(AUTOTUNE)

        logger.debug("dataset " + str(dataset))

        return dataset

    def get_raw_text_dataset(self):
        line_dataset = tf.data.experimental.CsvDataset(
            self.get_path(), [tf.string, tf.string])

        if self.has_maximum_size():
            line_dataset = line_dataset.take(self.get_maximum_size())

        line_dataset = line_dataset.cache(filename=self.get_text_cache_path())

        text_dataset = line_dataset.map(lambda x, y : y)

        return text_dataset

    def get_raw_text_generator(self):

        iterator = iter(self.get_raw_text_dataset())

        while True:
            try:
                x = next(iterator).numpy()
                yield x
            except StopIteration:
                return

    def load_file(self, path, label):
        wav_file = tf.io.read_file(path)

        audio_samples, audio_sample_rate = tf.audio.decode_wav(wav_file)

        audio_sample_count = tf.shape(audio_samples)[0]

        logger.debug("audio_samples " + str(audio_samples))
        logger.debug("audio_sample_count " + str(audio_sample_count))
        logger.debug("audio_sample_rate " + str(audio_sample_rate))
        logger.debug("label " + str(label))

        # TODO: use both channels
        sample = (audio_samples[:,0], audio_sample_count, audio_sample_rate, label), 0.0

        return sample

    def get_path(self):
        return self.source_config['path']

    def get_maximum_size(self):
        return int(self.source_config["maximum-size"])

    def has_maximum_size(self):
        return "maximum-size" in self.source_config

    def get_shuffle_window_size(self):
        return int(self.config['model']['shuffle-window-size'])

    def get_mini_batch_size(self):
        return int(self.config['model']['batch-size'])

    def group_by_sequence_length(self, dataset):

        def get_length(x,y):
            return x[1]

        boundaries = self.get_bucket_boundaries()

        bucket_transformation = tf.data.experimental.bucket_by_sequence_length(
            element_length_func = get_length,
            bucket_boundaries = boundaries,
            bucket_batch_sizes = [self.get_mini_batch_size() for i in range(len(boundaries) + 1)],
            padded_shapes=None,
            padding_values=None,
            pad_to_bucket_boundary=False,
            no_padding=True,
            drop_remainder=True)

        return dataset.apply(bucket_transformation)

    def get_bucket_boundaries(self):
        base = 2

        bucket_count = int(math.ceil(math.log(self.get_maximum_sequence_length(), base)))

        return [base ** i for i in range(1, bucket_count)]

    def get_maximum_sequence_length(self):
        return int(self.config['model']['maximum-sequence-length'])

    def get_cache_path(self):
        relative_path = self.get_relative_path(self.source_config["path"])

        path = os.path.join(self.config["system"]["cache"], relative_path)

        logger.debug("Cache file path is: " + path)
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        return path

    def get_text_cache_path(self):
        relative_path = self.get_relative_path(self.source_config["path"], modifier="-text")

        path = os.path.join(self.config["system"]["cache"], relative_path)

        logger.debug("Cache file path is: " + path)
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        return path

    def get_relative_path(self, path, modifier=""):
        path = os.path.splitdrive(path)[1]

        if path.find("s3://") == 0:
            path = path[5:]

        path = os.path.splitext(path)[0]

        return path + modifier + ".cache"


