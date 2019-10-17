
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class DataSources:
    def __init__(self, config):
        self.sources = []
        self.config = config

    def get_tf_dataset(self):
        dataset = self.sources[0].get_tf_dataset()

        if len(self.sources) > 1:
            for source in self.sources[1:]:
                dataset = dataset.concatenate(source.get_tf_dataset())

        dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        dataset = dataset.batch(self.get_mini_batch_size())

        if self.get_data_shuffle_window() > 0:
            dataset = dataset.shuffle(self.get_data_shuffle_window())

        return dataset

    def add_source(self, source):
        self.sources.append(source)

    def get_mini_batch_size(self):
        return int(self.config['model']['mini-batch-size'])

    def get_data_shuffle_window(self):
        return int(self.config['model']['data-shuffle-window'])


