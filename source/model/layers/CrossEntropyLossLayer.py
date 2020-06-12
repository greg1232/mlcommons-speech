
import tensorflow as tf
import tensorflow_addons as tfa

import logging

logger = logging.getLogger(__name__)

class CrossEntropyLossLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(CrossEntropyLossLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, mask=None):
        predictions, labels, label_lengths = inputs

        shifted_labels = labels - 1

        shifted_labels = tf.pad(shifted_labels, [[0,0], [0, tf.shape(predictions)[1]-tf.shape(labels)[1]]], constant_values=-1)


        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.maximum(shifted_labels, 0), predictions, from_logits=True,
        )

        cross_entropy_loss = self.mask_loss(cross_entropy_loss, shifted_labels)

        self.add_metric(cross_entropy_loss, name='cross_entropy_loss', aggregation='mean')

        #tf.print("labels", shifted_labels)
        #tf.print("loss", cross_entropy_loss)

        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, axis=-1)

        return cross_entropy_loss

    def mask_loss(self, loss, labels):
        # labels is [batch_size, sequence_length]
        # loss is [batch_size, sequence_length]

        mask = tf.cast(labels >= 0, dtype=tf.float32)

        return loss * mask

