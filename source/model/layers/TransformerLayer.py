import tensorflow as tf

from model.layers.MultiHeadedAttentionLayer import MultiHeadedAttentionLayer

class TransformerLayer:
    def __init__(self, config, causal=False):
        self.config = config

        self.layers = [MultiHeadedAttentionLayer(self.config, causal=causal) for
            layer in range(self.get_layer_count())]

    def __call__(self, inputs):

        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def get_layer_count(self):
        return int(self.config["model"]["layer-count"])






