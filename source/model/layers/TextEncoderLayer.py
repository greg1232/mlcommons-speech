import os

import tensorflow as tf
import tensorflow_datasets as tfds

import logging

logger = logging.getLogger(__name__)

class TextEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, training_dataset, **kwargs):
        super(TextEncoderLayer, self).__init__(dtype=tf.int64, trainable=False, **kwargs)
        self.config = config
        self.training_dataset = training_dataset

        self.init()

    def init(self):
        with tf.device('/cpu:0'):
            if not self.does_vocab_file_exist():
                logger.debug("Building vocab from corpus...")
                self.encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    self.training_dataset.get_raw_text_generator(),
                    self.get_target_vocab_size(),
                    max_corpus_chars = self.get_maximum_corpus_size_for_vocab(),
                    max_subword_length = self.get_maximum_subword_length())
                logger.debug(" Finished...")
                self.encoder.save_to_file(self.get_vocab_path())

            self.encoder = tfds.features.text.SubwordTextEncoder.load_from_file(
                self.get_vocab_path())

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        # output of this layer is a Tensor with dimensions [batch_size, max_sequence_length] of int64s (which are tokens)
        with tf.device('/cpu:0'):

            def encode(strings):

                # get strings
                encoded = [x.numpy().decode('utf8') for x in strings]

                # encode
                encoded = [self.encoder.encode(x) for x in encoded]

                # truncate
                lengths = [len(x) + 2 for x in encoded]

                max_lengths = [min(length, self.get_maximum_sequence_length()) for length in lengths]
                encoded = [x[0:max_length-2] for x, max_length in zip(encoded, max_lengths)]

                # pad
                max_length = max(max_lengths)
                encoded = [[self.get_document_start_token()] + x + [self.get_document_end_token()] + [0 for i in range(max_length - len(x))] for x in encoded]

                # convert to tensors
                encoded = tf.convert_to_tensor(encoded, dtype=tf.int64)
                lengths = tf.convert_to_tensor(lengths, dtype=tf.int64)

                lengths = [[length] for length in lengths]

                return encoded, lengths

            encoded, lengths = tf.py_function(encode, [inputs], [tf.int64, tf.int64])

            encoded_text = encoded[:,:-1]
            target_encoded_text = encoded[:, 1:]

            label_lengths = lengths - 1

            logger.debug("encoded text " + str(encoded_text))
            logger.debug("target encoded text " + str(target_encoded_text))
            logger.debug("label lengths " + str(label_lengths))

            return encoded_text, target_encoded_text, label_lengths

    def get_vocab_path(self):
        return os.path.join(self.config['model']['directory'], 'vocab')

    def get_target_vocab_size(self):
        return int(self.config['model']['vocab-size'])

    def get_vocab_size(self):
        return self.encoder.vocab_size

    def get_document_start_token(self):
        return self.get_vocab_size()

    def get_document_end_token(self):
        return self.get_document_start_token() + 1

    def get_total_vocab_size(self):
        return self.get_document_end_token() + 1

    def does_vocab_file_exist(self):
        return os.path.exists(self.get_vocab_path() + ".subwords")

    def get_maximum_sequence_length(self):
        return int(self.config['model']['maximum-sequence-length'])

    def get_maximum_subword_length(self):
        return int(self.config['model']['maximum-subword-length'])

    def get_maximum_corpus_size_for_vocab(self):
        return int(self.config['model']['maximum-corpus-size-for-vocab'])



