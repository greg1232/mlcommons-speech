
import logging

logger = logging.getLogger(__name__)

from model.AcousticModelFactory import AcousticModelFactory

class BeamSearchDecoder:
    def __init__(self, config, test_dataset):
        self.config = config
        self.test_dataset = test_dataset

        self.model = ModelFactory(self.config).create()

    def predict(self):
        for batch, label_batch in self.test_dataset.get_tensorflow_dataset():

            batch_size = batch[0].numpy().shape[0]
            for sample in range(batch_size):

                beam = [{"label" : "", "log-probability" : 0.0}]
                best_finished_label = None

                while len(beam) > 0:
                    self.expand_beam(beam, batch, sample)

                    best_finished_label = self.prune_beam(beam, best_finished_label)

                self.write_result(best_finished_label, batch, sample)

    def expand_beam(self, beam, batch, sample):
        new_beam = []

        for beam_entry in beam:
            batch_slice = self.get_batch_slice(batch, beam_entry["label"], sample)
            predicted_labels, probabilities = self.model.predict_on_batch(batch_slice, beam_size=self.get_beam_expansion_size())

            new_beam += [{"label" : label, "log-probability" : beam_entry["log-probability"] + probability} for label, probability in zip(predicted_labels[0], probabilities[0])]

        beam[:] = new_beam[:]

        logger.debug("Expanded beam: " + str(beam))

    def prune_beam(self, beam, best_finished_label):
        remaining_beam = []

        for beam_entry in beam:
            if self.is_finished(beam_entry["label"]):
                if best_finished_label is None or beam_entry["log-probability"] > best_finished_label["log-probability"]:
                    best_finished_label = beam_entry

            else:
                remaining_beam.append(beam_entry)

        sorted_beam = list(reversed(sorted(remaining_beam, key=lambda x : x["log-probability"])))
        if len(sorted_beam) > self.get_beam_size():
            sorted_beam = sorted_beam[:self.get_beam_size()]

        beam[:] = sorted_beam[:]

        logger.debug("Pruned beam: " + str(beam))

        if len(beam) == 0:
            return best_finished_label

        return beam[0]

    def is_finished(self, predicted_label):
        return predicted_label.find("<END>") != -1

    def get_batch_slice(self, batch, label, sample):
        audio_samples = batch[0][sample:sample+1, :]
        audio_sample_counts = batch[1][sample:sample+1]
        audio_sample_rates = batch[2][sample:sample+1]

        return (audio_samples, audio_sample_counts, audio_sample_rates, [label.strip("<START>")])

    def write_result(self, predicted_label, batch, sample):
        print(batch[3][sample].numpy().decode('utf-8'), predicted_label)

    def get_beam_expansion_size(self):
        return int(self.config["predictor"]["beam-expansion-size"])

    def get_beam_size(self):
        return int(self.config["predictor"]["beam-size"])


