
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

        dataset = dataset.take(self.get_maximum_dataset_size())

        dataset = dataset.cache(self.get_cache_file())

        dataset = dataset.padded_batch(self.get_mini_batch_size(),
            padded_shapes=((None, 1), (), (), ()))

        if self.get_data_shuffle_window() > 0:
            dataset = dataset.shuffle(self.get_data_shuffle_window())

        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset

    def add_source(self, source):
        self.sources.append(source)

    def get_mini_batch_size(self):
        return int(self.config['model']['mini-batch-size'])

    def get_data_shuffle_window(self):
        return int(self.config['model']['data-shuffle-window'])

    def get_maximum_dataset_size(self):
        return int(self.config['model']['maximum-dataset-size'])

    def get_cache_file(self):
        return self.sources[0].get_cache_file()



