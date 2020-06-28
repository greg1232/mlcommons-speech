import os

import tensorflow as tf

from model.layers.TextEncoderLayer import TextEncoderLayer
from model.layers.TransformerLayer import TransformerLayer
from model.layers.DummyLoss import DummyLoss
from model.layers.CrossEntropyLossLayer import CrossEntropyLossLayer

import logging

logger = logging.getLogger(__name__)

class TransformerLanguageModel:
    def __init__(self, config, training_dataset, validation_dataset):
        self.config = config
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.create_or_load_model()

    def train(self):

        self.training_model.fit(x=self.training_dataset.get_tensorflow_dataset(),
            validation_data=self.validation_dataset.get_tensorflow_dataset(),
            validation_steps=self.get_validation_steps(),
            epochs=self.get_epochs(),
            callbacks=self.get_callbacks())

        self.checkpoint()

    def get_callbacks(self):
        return [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            tf.keras.callbacks.EarlyStopping(
                patience=self.get_early_stopping_patience(), mode='min', monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(
                self.get_best_model_directory(), mode='min',
                save_best_only=self.get_save_best_only(),
                verbose=1, save_weights_only=True,
                save_freq=self.get_save_frequency(),
                monitor='val_loss'),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config['language-model']['directory'], 'logs'),
                profile_batch=self.get_profile_batch(),
                update_freq=100)
        ]

    def create_model(self):
        labels = tf.keras.Input(shape=(), dtype=tf.string)

        label_indices, target_label_indices, label_lengths = self.encode_text(labels)

        decoder_model_features = self.run_transformer_decoder(label_indices, label_lengths)

        decoder_model_outputs = tf.keras.layers.Dense(self.get_vocab_size())(decoder_model_features)

        cross_entropy_loss = CrossEntropyLossLayer(self.config)(
            [decoder_model_outputs, target_label_indices, label_lengths])

        self.training_model = tf.keras.Model(
            inputs=labels,
            outputs=cross_entropy_loss)

        self.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(self.get_learning_rate()),
            loss=DummyLoss())

        print(self.training_model.summary())
        print(self.training_model.metrics_names)

        token_probabilities = tf.keras.layers.Softmax()(decoder_model_outputs)

        self.inference_model = tf.keras.Model(
            inputs=[labels],
            outputs=[token_probabilities, label_lengths])

    def run_transformer_decoder(self, labels, label_lengths):

        labels = tf.keras.layers.Reshape((-1,))(labels)
        logger.debug("transformer decoder labels: " + str(labels))

        label_embedding = tf.keras.layers.Embedding(
            self.get_vocab_size(), self.get_layer_size(), mask_zero=True)(labels)

        hidden = label_embedding

        return TransformerLayer(self.config, self.config["language-model"], causal=True)(hidden)

    def encode_text(self, labels):
        self.text_encoder_layer = TextEncoderLayer(
            self.config, self.config["language-model"], self.training_dataset)

        return self.text_encoder_layer(labels)

    def get_vocab_size(self):
        return self.text_encoder_layer.get_total_vocab_size()

    def get_document_end_token(self):
        return self.text_encoder_layer.get_document_end_token()

    def checkpoint(self):
        self.training_model.save_weights(self.get_checkpoint_model_directory())

    def create_or_load_model(self):

        logger.debug("Loading or creating model from directory: " +
            self.config['language-model']['directory'])

        self.create_model()

        if self.does_model_exist():
            self.load_model()

    def does_model_exist(self):
        if os.path.exists(self.get_best_model_directory()):
            return True

        if os.path.exists(self.get_checkpoint_model_directory()):
            return True

        return False

    def load_model(self):
        path = self.get_checkpoint_model_directory()

        if os.path.exists(self.get_best_model_directory()):
            path = self.get_best_model_directory()

        self.training_model.load_weights(path, by_name=True)
        logger.debug("Loading model from : " + path)

    def get_layer_size(self):
        return int(self.config['language-model']['layer-size'])

    def get_epochs(self):
        return int(self.config['language-model']['epochs'])

    def get_learning_rate(self):
        return float(self.config['language-model']['learning-rate'])

    def get_profile_batch(self):
        should_profile = str(self.config["language-model"]["enable-profiler"]
            ).lower() in ['true', '1']

        if should_profile:
            return 3
        else:
            return 0

    def get_model_directory(self):
        return self.config['language-model']['directory']

    def get_best_model_directory(self):
        return os.path.join(self.get_model_directory(), 'best.h5')

    def get_checkpoint_model_directory(self):
        return os.path.join(self.get_model_directory(), 'checkpoint.h5')

    def get_early_stopping_patience(self):
        return int(self.config['language-model']['early-stopping-patience'])

    def get_validation_steps(self):
        if not 'validation-steps' in self.config['language-model']:
            return None

        return int(self.config['language-model']['validation-steps'])

    def get_save_frequency(self):
        try:
            return int(self.config['language-model']['save-frequency'])
        except KeyError:
            return 'epoch'

    def get_save_best_only(self):
        return self.get_save_frequency() == 'epoch'

    def get_maximum_sequence_length(self):
        return int(self.config['language-model']['maximum-sequence-length'])









