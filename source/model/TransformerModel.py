
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import logging

logger = logging.getLogger(__name__)

class TransformerModel:
    def __init__(self, config, training_dataset, validation_dataset):
        self.config = config
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.create_or_load_model()

    def train(self):

        callbacks = [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            #tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(self.get_best_model_directory(),
                save_best_only=True, save_weights_only=True,
                monitor='val_accuracy'),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config['model']['directory'], 'logs'))
        ]

        with tf.device('/cpu:0'):
            self.model.fit(
                x=self.encode_dataset(self.training_dataset.get_tensorflow_dataset()),
                validation_data=self.encode_dataset(self.validation_dataset.get_tensorflow_dataset()),
                epochs=self.get_epochs(),
                callbacks=callbacks)

        self.checkpoint()

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def checkpoint(self):
        self.model.save_weights(self.get_checkpoint_model_directory())

    def create_or_load_model(self):

        logger.debug("Loading or creating model from directory: " +
            self.config['model']['directory'])

        self.create_encoder()

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

        self.model.load_weights(path)

    def create_model(self):
        inputs = tf.keras.Input(shape=(None, self.get_frame_size()), dtype=tf.float32)

        hidden = tf.keras.layers.Reshape((-1, 4, self.get_frame_size()//4))(inputs)

        hidden = self.add_conv_layer(hidden)
        hidden = self.add_attenion_layer(hidden)

        outputs = tf.keras.layers.Dense(self.get_vocab_size() + 1)(hidden)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(self.getLearningRate()),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        print(model.summary())

        self.model = model

    def add_conv_layer(self, hidden):
        inputs = tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(filters=self.get_layer_size(), kernel_size=3, strides=1,
            padding='same', activation='relu')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        return tf.keras.layers.Add()([inputs, hidden])

    def add_attenion_layer(self, hidden):

        query = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')(hidden)
        value = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')(hidden)
        updated = tf.keras.layers.Attention(use_scale=True)([query, value])
        updated = tf.keras.layers.BatchNormalization()(updated)

        result = tf.keras.layers.Add()([updated, hidden])

        return result


    def create_encoder(self):
        labels = self.training_dataset.get_tf_dataset().map(lambda x, y : y)

        if not self.does_vocab_file_exist():
            logger.debug("Building vocab from corpus...")
            raw_label_text_dataset = labels.unbatch()

            def to_string(dataset):
                for tensor in iter(dataset):
                    label = str(tensor.numpy())

                    logger.debug(label)

                    yield label

            self.encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                to_string(raw_label_text_dataset), self.get_target_vocab_size())
            logger.debug(" Finished...")
            self.encoder.save_to_file(self.get_vocab_path())

        self.encoder = tfds.features.text.SubwordTextEncoder.load_from_file(
            self.get_vocab_path())

    def encode_dataset(self, dataset):

        audio  = dataset.map(lambda x, y : x)
        labels = dataset.map(lambda x, y : y)

        encoded_audio  = self.encode_audio(audio)
        encoded_labels = self.encode_labels(labels)

        return tf.Dataset.zip((encoded_audio, encoded_labels))

    def encode_labels(self, labels):
        with tf.device('/cpu:0'):

            def encode_label(x):

                encoded = [self.encoder.encode(str(t.numpy()[0])) for t in x]

                max_length = max([len(x) for x in encoded])

                # pad
                encoded = [x + [self.encoder.vocab_size for i in
                    range(max_length - len(x))] for x in encoded]

                encoded = tf.convert_to_tensor(encoded, dtype=tf.int64)

                return encoded

            return labels.map(encode_label)

    def encode_audio(self, audio):

        def encode_audio_batch(wave_batch):
            framed_batch = tf.signal.frame(wave_batch,
                self.get_frame_length(),
                self.get_frame_step(),
                pad_end=True)

            frequencies = tf.signal.fft(framed_batch)

            return frequencies

        return audio.map(encode_audio_batch)

    def get_layer_size(self):
        return int(self.config['model']['layer-size'])

    def get_epochs(self):
        return int(self.config['model']['epochs'])

    def get_learning_rate(self):
        return float(self.config['model']['learning-rate'])

    def get_model_directory(self):
        return self.config['model']['directory']

    def get_best_model_directory(self):
        return os.path.join(self.get_model_directory(), 'best.h5')

    def get_checkpoint_model_directory(self):
        return os.path.join(self.get_model_directory(), 'checkpoint.h5')

    def get_vocab_path(self):
        return os.path.join(self.get_model_directory(), 'vocab')

    def get_target_vocab_size(self):
        return int(self.config['model']['vocab-size'])

    def get_vocab_size(self):
        return self.encoder.vocab_size

    def does_vocab_file_exist(self):
        return os.path.exists(self.get_vocab_path() + ".subwords")

    def get_frame_size(self):
        return int(self.config['model']['frame-size'])

    def get_frame_step(self):
        return int(self.config['model']['frame-step'])








