
from google.cloud import storage
from argparse import ArgumentParser
import logging

logger = logging.getLogger(__name__)

from smart_open import open

def make_splits(arguments):
    samples = []

    get_common_voice_samples(samples)
    get_librispeech_samples(samples)
    get_librivox_samples(samples)
    #get_voicery_samples(samples)
    #get_cc_search_samples(samples)

    train, test, development = split_samples(arguments, samples)

    save_samples(train, arguments["train_path"])
    save_samples(test, arguments["test_path"])
    save_samples(development, arguments["development_path"])

def get_common_voice_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/common-voice/train.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/common-voice/test.csv")

def load_csv_samples(samples, csv_path):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in reader:
            path, caption

            metadata = {}
            if len(row) >= 3:
                metadata = json.loads(row[2])

            samples.append({"path" : path, "caption" : catpion, "metadata" : metadata})

def get_librispeech_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/dev-clean.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/dev-other.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/test-clean.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/test-other.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-360.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-other-500.csv")

def get_librivox_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librivox-v0.1/data.csv")

def get_voicery_samples(samples):
    assert False, "Not implemented."

def get_cc_search_samples(samples):
    extract_aligned_samples(samples, "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020", "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020_aligned_data_9_15_20")

def extract_alignd_samples(samples, audio_path, alignment_path):

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(path)

    for blob in blobs:
        if blob.name

def split_samples(arguments, samples):

    id_map = {}

    for sample in samples:
        sample_id = get_id_for_sample(id_map, sample)

        if not sample_id in id_map:
            id_map[sample_id] = []

        id_map[sample_id].append(sample)

    ids = stable_shuffle(id_map)

    test_set_size = min(len(samples) // 3, int(arguments["test_set_size"]))

    test = extract_samples(ids, test_set_size)
    development = extract_samples(ids, test_set_size)

    train = extract_samples(ids, len(samples))

    return train, test, development

def get_id_for_sample(id_map, sample):
    return len(id_map)

def stable_shuffle(id_map):
    id_list = [(key, value) for key, value in id_map.items()]

    generator = random.Random(seed=42)

    generator.shuffle(id_list)

    return id_list

def extract_samples(ids, count):
    sample_count = 0
    id_count = 0

    for index, samples in ids:
        sample_count += len(samples)
        id_count += 1

        if sample_count >= count:
            break

    new_samples = []

    for index, samples in ids[:id_count]:
        new_samples.extend(samples)

    del ids[:id_count]

    return new_samples

def save_samples(samples, path):
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')

        for sample in samples:
            writer.write_row([sample["path"], sample["caption"], json.dumps(sample["metadata"])])

def main():
    parser = ArgumentParser("Creates people's speech train, test, "
        "development v0.5 splits.")

    parser.add_argument("--train-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.5/train.csv",
        help = "The output path to save the training dataset.")
    parser.add_argument("--test-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.5/test.csv",
        help = "The output path to save the test dataset.")
    parser.add_argument("--development-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.5/development.csv",
        help = "The output path to save the development dataset.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    make_splits(arguments)

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



