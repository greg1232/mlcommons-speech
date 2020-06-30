import os

import tensorflow as tf
import tensorflow_datasets as tfds

import logging

logger = logging.getLogger(__name__)

class AudioEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(AudioEncoderLayer, self).__init__(dtype=tf.int64, trainable=False, **kwargs)
        self.config = config

    def call(self, inputs):
        audio_samples, audio_sample_counts, audio_sampling_rate = inputs

        frequencies = tf.signal.stft(audio_samples,
            self.get_frame_size(),
            self.get_frame_step(),
            pad_end=True)

        real_frequencies = tf.dtypes.cast(frequencies, tf.float32)

        logger.debug("frequencies " + str(real_frequencies))

        #tf.print("frequencies", tf.shape(real_frequencies))

        # TODO: add MEL

        # TODO: add spec augment: https://arxiv.org/pdf/1904.08779.pdf

        return real_frequencies, (audio_sample_counts + self.get_frame_step() - 1) // self.get_frame_step()

    def get_frame_size(self):
        return int(self.config['acoustic-model']['frame-size'])

    def get_frame_step(self):
        return int(self.config['acoustic-model']['frame-step'])

