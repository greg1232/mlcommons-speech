
import tensorflow as tf

import logging

logger = logging.getLogger(__name__)

class MultiHeadedAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config, causal=False, **kwargs):
        super(MultiHeadedAttentionLayer, self).__init__(**kwargs)
        self.config = config

        self.norm_1 = tf.keras.layers.BatchNormalization()
        self.dense_1 = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')
        self.attention = tf.keras.layers.Attention(use_scale=True, causal=causal,
            dropout=self.get_dropout())
        self.dropout_1 = tf.keras.layers.Dropout(self.get_dropout())

        self.add_1 = tf.keras.layers.Add()

        self.norm_2 = tf.keras.layers.BatchNormalization()
        self.dense_3 = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')
        self.dense_4 = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')
        self.dense_5 = tf.keras.layers.Dense(self.get_layer_size())

        self.dropout_2 = tf.keras.layers.Dropout(self.get_dropout())
        self.dropout_3 = tf.keras.layers.Dropout(self.get_dropout())

        self.add_2 = tf.keras.layers.Add()

        assert self.get_layer_size() % self.get_attention_head_count() == 0

    def call(self, inputs, mask=None):

        # attention processing
        logger.debug("attention-inputs: " + str(inputs))
        logger.debug("mask: " + str(mask))

        updated = self.norm_1(inputs)
        logger.debug("norm: " + str(updated))

        query = self.dense_1(updated)
        value = self.dense_2(updated)
        logger.debug("query: " + str(query))
        logger.debug("value: " + str(value))

        split_out_query_heads = self.split_out_heads(query)
        split_out_value_heads = self.split_out_heads(value)

        updated = self.attention([split_out_query_heads, split_out_value_heads])
        logger.debug("attention: " + str(updated))
        updated = self.dropout_1(updated)
        logger.debug("dropout-result: " + str(updated))

        updated = self.merge_heads(updated)

        updated = self.dense_3(updated)
        updated = self.dropout_2(updated)

        attention_result = self.add_1([inputs, updated])

        # linear processing
        updated = self.norm_2(attention_result)

        updated = self.dense_4(updated)
        updated = self.dropout_3(updated)
        updated = self.dense_5(updated)

        updated = self.add_2([attention_result, updated])

        return updated

    def compute_mask(self, inputs, mask=None):

        return mask

    def merge_heads(self, inputs):
        attention_heads = self.get_attention_head_count()
        head_layer_size = self.get_layer_size() // self.get_attention_head_count()
        timesteps = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0] // self.get_attention_head_count()

        split_out_head_results = tf.reshape(inputs, (batch_size, attention_heads, timesteps, head_layer_size))
        transposed_back_heads = tf.transpose(split_out_head_results, perm=[0, 2, 1, 3])
        result = tf.reshape(transposed_back_heads, (batch_size, timesteps, self.get_layer_size()))

        return result

    def split_out_heads(self, inputs):
        attention_heads = self.get_attention_head_count()
        head_layer_size = self.get_layer_size() // self.get_attention_head_count()
        timesteps = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]

        split_out_heads = tf.reshape(inputs, (batch_size, timesteps, attention_heads, head_layer_size))
        transposed_heads = tf.transpose(split_out_heads, perm=[0, 2, 1, 3])
        collapsed_heads_into_batch = tf.reshape(transposed_heads, (batch_size * attention_heads, timesteps, head_layer_size))

        return collapsed_heads_into_batch

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_layer_size(self):
        return int(self.config["model"]["layer-size"])

    def get_attention_head_count(self):
        return int(self.config["model"]["attention-head-count"])

    def get_dropout(self):
        return float(self.config["model"]["dropout"])





