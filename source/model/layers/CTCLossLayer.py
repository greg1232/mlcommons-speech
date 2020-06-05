
import tensorflow as tf
import tensorflow_addons as tfa

import logging

logger = logging.getLogger(__name__)

class CTCLossLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(CTCLossLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, mask=None):
        predictions, input_lengths, labels, label_lengths = inputs

        #tf.print("labels shape ", tf.shape(labels))
        #tf.print("predictions shape ", tf.shape(predictions))
        #tf.print("input_lengths", input_lengths)
        #tf.print("label_lengths", label_lengths)

        labels = tf.pad(labels, [[0,0], [0, tf.shape(predictions)[1]-tf.shape(labels)[1]]])

        ctc_loss = tf.keras.backend.ctc_batch_cost(
                        labels,
                        predictions,
                        input_lengths,
                        label_lengths
                    )

        self.add_metric(ctc_loss, name='ctc_loss', aggregation='mean')

        ctc_loss = tf.reduce_mean(ctc_loss)

        return ctc_loss

