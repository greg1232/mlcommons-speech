from argparse import ArgumentParser
import logging
import csv
import os
import json
from smart_open import open
from cleantext import clean

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("Add metadata to librispeech data.")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100.csv",
        help = "The output path to load the dataset from.")
    parser.add_argument("-m", "--metadata-path", default = "gs://the-peoples-speech-aws-import/librispeech-formatted/SPEAKERS.txt",
        help = "The path to load the metadata from.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100-metadata.csv",
        help = "The output path to save dataset with metadata.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    add_metadata(arguments)

def add_metadata(arguments):
    samples = load_csv(arguments["input_path"])
    metadata = load_metadata(arguments["metadata_path"])

    updated_samples = update_samples(samples, metadata)

    with open(arguments["output_path"], "w", newline="") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')
        for sample in updated_samples:
            csv_writer.writerow(sample)

def decomment(csvfile):
    for row in csvfile:
        raw = row.split(';')[0].strip()
        if raw: yield raw

def load_metadata(speakers_path):
    with open(speakers_path) as speakers_file:
        csv_reader = csv.reader(decomment(speakers_file), delimiter='|', quotechar='"')

def load_csv(csv_path):
    new_samples = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in reader:
            path, caption = row[0], row[1]

            metadata = {}
            if len(row) >= 3:
                if len(row[2]) > 0:
                    metadata = json.loads(row[2])

            yield {"path" : path, "caption" : caption, "metadata" : metadata}

def update_samples(samples, metadata):
    for sample in samples:
        name = get_speaker_id_for_sample(sample["path"])
        metadata = metadata[name]
        logger.debug("For " + sample["path"])
        logger.debug("Added metadata " + str(metadata))
        yield (sample["path"], sample["caption"], metadata)

def setup_logger(arguments):

   if arguments["verbose"]:
       logger.setLevel(logging.DEBUG)
   else:
       logger.setLevel(logging.INFO)

   ch = logging.StreamHandler()
   ch.setLevel(logging.DEBUG)

   # create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

   # add formatter to ch
   ch.setFormatter(formatter)

   # add ch to logger
   logger.addHandler(ch)

if __name__ == "__main__":
    main()




