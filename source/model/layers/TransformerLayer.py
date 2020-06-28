import tensorflow as tf

from model.layers.MultiHeadedAttentionLayer import MultiHeadedAttentionLayer

class TransformerLayer:
    def __init__(self, config, model_config, causal=False):
        self.config = config
        self.model_config = model_config

        self.layers = [MultiHeadedAttentionLayer(self.config, model_config, causal=causal) for
            layer in range(self.get_layer_count())]

    def __call__(self, inputs):

        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def get_layer_count(self):
        return int(self.model_config["layer-count"])






