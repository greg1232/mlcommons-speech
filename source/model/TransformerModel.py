
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import logging

logger = logging.getLogger(__name__)

class TransformerModel:
    def __init__(self, config, training_dataset, validation_dataset):
        self.config = config
        self.training_dataset = training_dataset.get_tf_dataset()
        self.validation_dataset = validation_dataset.get_tf_dataset()

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

        self.model.fit(
            x=self.encode_dataset(self.training_dataset),
            validation_data=self.encode_dataset(self.validation_dataset),
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
        inputs = tf.keras.Input(shape=(None, self.get_frame_size() // 2 + 1), dtype=tf.float32)
        label = tf.keras.Input(shape=(None,), dtype=tf.int64)
        input_length = tf.keras.Input(shape=(1), dtype=tf.int64)
        label_length = tf.keras.Input(shape=(1), dtype=tf.int64)

        hidden = self.run_encoder(inputs)
        transducer = self.run_decoder(hidden, label, label_length)

        outputs = tf.keras.layers.Dense(self.get_vocab_size() + 1,
            activation='softmax')(hidden)

        ctc_loss = self.compute_ctc_loss(outputs, input_length, label, label_length)

        model = tf.keras.Model(
            inputs=[inputs, label, input_length, label_length],
            outputs=ctc_loss)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.get_learning_rate()),
            loss=lambda y_true, y_pred : y_pred)

        print(model.summary())

        self.model = model

    def run_encoder(self, inputs):

        hidden = self.add_input_conv_layer(inputs)

        hidden = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')(hidden)

        hidden = self.add_conv_layer(hidden)
        hidden = self.add_attention_layer(hidden)

        return hidden

    def run_decoder(self, hidden, labels, label_lengths):

        def shift(labels):
            initial_label = tf.cast(tf.fill([tf.shape(labels)[0], 1], self.get_vocab_size()), tf.int64)
            return tf.concat([initial_label, labels[:,:-1]], axis=1)

        shifted_labels = tf.keras.layers.Lambda(shift)(labels)

        label_embedding = tf.keras.layers.Embedding(
            self.get_vocab_size() + 1, self.get_layer_size())(shifted_labels)

        hidden = tf.keras.layers.Add()([label_embedding, hidden])

        hidden = self.add_attention_layer(hidden, causal=True)

        return hidden

    def compute_ctc_loss(self, predictions, input_lengths, labels, label_lengths):
        def loss(args):
            predictions, input_lengths, labels, label_lengths = args

            return tf.keras.backend.ctc_batch_cost(
                        labels,
                        predictions,
                        input_lengths,
                        label_lengths
                    )
        return tf.keras.layers.Lambda(loss, output_shape=(1,), name="ctc")(
            [predictions, input_lengths, labels, label_lengths])


    def add_conv_layer(self, hidden):

        inputs = tf.keras.layers.MaxPool1D(pool_size=1, strides=1, padding='same')(hidden)
        hidden = tf.keras.layers.Conv1D(filters=self.get_layer_size(), kernel_size=3, strides=1,
            padding='same', activation='relu')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)

        return tf.keras.layers.Add()([inputs, hidden])


    def add_input_conv_layer(self, hidden):
        batch = tf.shape(hidden)[0]
        timesteps = tf.shape(hidden)[1]
        features = hidden.shape[2]

        new_shape    = [batch, timesteps, features, 1]
        result_shape = [batch, timesteps, features * self.get_layer_size()]

        def reshape(x):
            return tf.reshape(x, new_shape)

        hidden = tf.keras.layers.Lambda(reshape)(hidden)

        inputs = tf.keras.layers.MaxPool2D(pool_size=1, strides=1, padding='same')(hidden)
        hidden = tf.keras.layers.Conv2D(filters=self.get_layer_size(), kernel_size=3, strides=1,
            padding='same', activation='relu')(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)

        result = tf.keras.layers.Add()([inputs, hidden])

        def reshape_back(x):
            return tf.reshape(x, result_shape)

        return tf.keras.layers.Lambda(reshape_back)(result)

    def add_attention_layer(self, hidden, *, causal=False):

        query = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')(hidden)
        value = tf.keras.layers.Dense(self.get_layer_size(), activation='relu')(hidden)
        updated = tf.keras.layers.Attention(use_scale=True, causal=causal)([query, value])
        updated = tf.keras.layers.BatchNormalization()(updated)

        result = tf.keras.layers.Add()([updated, hidden])

        return result


    def create_encoder(self):

        if not self.does_vocab_file_exist():
            logger.debug("Building vocab from corpus...")
            raw_label_text_dataset = self.training_dataset.unbatch()

            raw_label_text_dataset = raw_label_text_dataset.map(
                lambda samples, sample_counts, sample_rates, labels : labels)

            def to_string(dataset):

                for sample in iter(dataset):
                    label = str(sample.numpy())

                    logger.debug(label)

                    yield label

            self.encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                to_string(raw_label_text_dataset), self.get_target_vocab_size())
            logger.debug(" Finished...")
            self.encoder.save_to_file(self.get_vocab_path())

        self.encoder = tfds.features.text.SubwordTextEncoder.load_from_file(
            self.get_vocab_path())

    def encode_dataset(self, dataset):

        audio = dataset.map(
            lambda samples, sample_counts, sample_rates, labels :
                (samples, sample_rates))

        labels = dataset.map(
            lambda samples, sample_counts, sample_rates, labels :
                (sample_counts, labels))

        encoded_audio  = self.encode_audio(audio)
        encoded_labels = self.encode_labels(labels)

        encoded_dataset = tf.data.Dataset.zip((encoded_audio, encoded_labels))
        encoded_dataset = encoded_dataset.map(lambda x, y:  ((x, y[0], y[1], y[2]), 0))

        return encoded_dataset

    def encode_labels(self, labels):
        with tf.device('/cpu:0'):

            def encode_label(input_lengths, label):

                encoded = [self.encoder.encode(str(t.numpy())) for t in label]

                label_lengths = [[len(x)] for x in encoded]
                input_lengths = [[x // self.get_frame_step()] for x in input_lengths]

                max_length = max([x[0] for x in label_lengths])

                # pad
                encoded = [x + [self.encoder.vocab_size for i in
                    range(max_length - len(x))] for x in encoded]

                return encoded, input_lengths, label_lengths

            return labels.map(lambda x, y : tf.py_function(encode_label, [x, y], [tf.int64, tf.int64, tf.int64]))

    def encode_audio(self, audio):

        def encode_audio_batch(wav_batch, sample_rate_batch):

            frequencies = tf.signal.stft(wav_batch[:,:,0],
                self.get_frame_size(),
                self.get_frame_step(),
                pad_end=True)

            real_frequencies = tf.dtypes.cast(frequencies, tf.float32)

            return real_frequencies

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








