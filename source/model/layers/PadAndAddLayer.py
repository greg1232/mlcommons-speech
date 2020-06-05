

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)

class PadAndAddLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PadAndAddLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):

        left, right = inputs

        logger.debug("pad right " + str(left))
        logger.debug("pad left " + str(right))

        left_shape  = tf.shape(left)[1]
        right_shape = tf.shape(right)[1]

        max_shape = tf.maximum(left_shape, right_shape)

        left_gap = max_shape - left_shape
        right_gap = max_shape - right_shape

        padded_left = tf.pad(left, [[0, 0], [0, left_gap], [0, 0]])
        padded_right = tf.pad(right, [[0, 0], [0, right_gap], [0, 0]])

        return padded_left + padded_right




