
import logging

logger = logging.getLogger(__name__)

from model.ModelFactory import ModelFactory

class GreedyDecoder:
    def __init__(self, config, test_dataset):
        self.config = config
        self.test_dataset = test_dataset

        self.model = ModelFactory(self.config).create()

    def predict(self):
        for batch, label_batch in self.test_dataset.get_tensorflow_dataset():

            batch_size = batch[0].numpy().shape[0]
            for sample in range(batch_size):
                predicted_label = ""
                while not self.is_finished(predicted_label):
                    batch_slice = self.get_batch_slice(batch, predicted_label, sample)
                    predicted_label = self.model.predict_on_batch(batch_slice)[0]
                    logger.debug("Label is: '" + predicted_label + "'")

                self.write_result(predicted_label, batch, sample)

    def is_finished(self, predicted_label):
        return predicted_label.find("<END>") != -1

    def get_batch_slice(self, batch, label, sample):
        audio_samples = batch[0][sample:sample+1, :]
        audio_sample_counts = batch[1][sample:sample+1]
        audio_sample_rates = batch[2][sample:sample+1]

        return (audio_samples, audio_sample_counts, audio_sample_rates, [label.strip("<START>")])

    def write_result(self, predicted_label, batch, sample):
        print(batch[3][sample].numpy().decode('utf-8'), predicted_label)

