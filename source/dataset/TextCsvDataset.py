
import tensorflow as tf
import tensorflow_io as tfio
import os
import math

AUTOTUNE = tf.data.experimental.AUTOTUNE

import logging

logger = logging.getLogger(__name__)

class TextCsvDataset:
    def __init__(self, config, source_config):
        self.config = config
        self.source_config = source_config

    def get_tensorflow_dataset(self):
        line_dataset = tf.data.experimental.CsvDataset(
            self.get_path(), [tf.string], select_cols=[1])

        if self.has_maximum_size():
            line_dataset = line_dataset.take(self.get_maximum_size())

        dataset = line_dataset
        dataset = dataset.map(lambda x: (x,0))
        dataset = dataset.shuffle(self.get_shuffle_window_size(), seed=42)
        dataset = self.group_by_sequence_length(dataset)

        dataset = dataset.prefetch(AUTOTUNE)

        logger.debug("dataset " + str(dataset))

        return dataset

    def get_raw_text_dataset(self):
        line_dataset = tf.data.experimental.CsvDataset(
            self.get_path(), [tf.string], select_cols=[0])

        if self.has_maximum_size():
            line_dataset = line_dataset.take(self.get_maximum_size())

        text_dataset = line_dataset

        return text_dataset

    def get_raw_text_generator(self):

        iterator = iter(self.get_raw_text_dataset())

        while True:
            try:
                x = next(iterator)[0].numpy()
                yield x
            except StopIteration:
                return

    def get_path(self):
        return self.source_config['path']

    def get_maximum_size(self):
        return int(self.source_config["maximum-size"])

    def has_maximum_size(self):
        return "maximum-size" in self.source_config

    def get_shuffle_window_size(self):
        return int(self.config['language-model']['shuffle-window-size'])

    def get_mini_batch_size(self):
        return int(self.config['language-model']['batch-size'])

    def group_by_sequence_length(self, dataset):

        def get_length(x, y):
            return tf.strings.length(x)

        boundaries = self.get_bucket_boundaries()

        bucket_transformation = tf.data.experimental.bucket_by_sequence_length(
            element_length_func = get_length,
            bucket_boundaries = boundaries,
            bucket_batch_sizes = [self.get_mini_batch_size() for i in range(len(boundaries) + 1)],
            padded_shapes=None,
            padding_values=None,
            pad_to_bucket_boundary=False,
            no_padding=False,
            drop_remainder=True)

        return dataset.apply(bucket_transformation)

    def get_bucket_boundaries(self):
        base = 2

        bucket_count = int(math.ceil(math.log(self.get_maximum_sequence_length(), base)))

        return [base ** i for i in range(1, bucket_count)]

    def get_maximum_sequence_length(self):
        return int(self.config['language-model']['maximum-sequence-length'])

