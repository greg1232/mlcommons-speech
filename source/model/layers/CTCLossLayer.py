
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
        shifted_labels = labels - 1

        shifted_labels = tf.pad(shifted_labels, [[0,0], [0, tf.shape(predictions)[1]-tf.shape(labels)[1]]], constant_values=-1)

        ctc_loss = tf.nn.ctc_loss(
                        tf.maximum(shifted_labels, 0),
                        predictions,
                        tf.reshape(label_lengths, (-1,)),
                        tf.reshape(input_lengths, (-1,)),
                        logits_time_major = False,
                        blank_index=-1
                    )
        #tf.print("labels", shifted_labels, summarize=-1)
        #tf.print("input_lengths", input_lengths)
        #tf.print("label_lengths", label_lengths)
        #tf.print("ctc_loss", ctc_loss, summarize=-1)

        self.add_metric(ctc_loss, name='ctc_loss', aggregation='mean')

        #ctc_loss = tf.reduce_mean(ctc_loss, axis=-1)

        ctc_loss *= self.get_ctc_loss_scale()

        return ctc_loss


    def get_ctc_loss_scale(self):
        return (self.config["model"]["ctc-loss-scale"])

