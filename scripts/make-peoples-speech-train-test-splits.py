
from google.cloud import storage
from argparse import ArgumentParser
import logging
import csv
import os
import json
import random
from pydub import AudioSegment

logger = logging.getLogger(__name__)

from smart_open import open

def make_splits(arguments):
    samples = []

    get_common_voice_samples(samples)
    get_librispeech_samples(samples)
    get_librivox_samples(samples)
    get_voicery_samples(samples)
    get_cc_search_samples(samples)

    train, test, development = split_samples(arguments, samples)

    save_samples(train, arguments["train_path"])
    save_samples(test, arguments["test_path"])
    save_samples(development, arguments["development_path"])

def get_common_voice_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/common-voice/train.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/common-voice/test.csv")

def load_csv_samples(samples, csv_path):
    new_samples = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in reader:
            path, caption = row[0], row[1]

            metadata = {}
            if len(row) >= 3:
                metadata = json.loads(row[2])

            new_samples.append({"path" : path, "caption" : caption, "metadata" : metadata})

    logger.info("Loaded " + str(len(new_samples)) + " from " + csv_path)

    samples.extend(new_samples)

def get_librispeech_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/dev-clean.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/dev-other.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/test-clean.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/test-other.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-360.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-other-500.csv")

def get_librivox_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librivox-v0.2-1M/data.csv")

def get_voicery_samples(samples):
    mp3_files = get_mp3_files("gs://the-peoples-speech-aws-import/voicery")

    for path, name in mp3_files.items():
        transcript = get_voicery_transcript(path)

        samples.append((path, transcript, {"speaker_id" : "voicery_" + name}))

storage_client = storage.Client()

def get_mp3_files(audio_path):
    logger.debug("Getting MP3 files under " + audio_path)

    # Note: Client.list_blobs requires at least package version 1.17.0.
    bucket_name, prefix = get_bucket_and_prefix(audio_path)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    mp3_files = {}

    for blob in blobs:
        if is_mp3(blob.name):
            path = os.path.join("gs://" + bucket_name, blob.name)
            mp3_files[path] = get_key(blob.name)

    logger.debug(" Found " + str(len(mp3_files)) + " mp3 files")

    return mp3_files

def get_key(path):
    parts = split_all(path)
    return os.path.splitext(parts[-2] + "-" + parts[-1])[0]

def is_mp3(path):
    return os.path.splitext(path)[1] == ".mp3" or os.path.splitext(path)[1] == ".wav"

def get_bucket_and_prefix(path):
    parts = split_all(path[5:])

    return parts[0], os.path.join(*parts[1:])

def split_all(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_key(path):
    parts = split_all(path)
    return os.path.splitext(parts[-2] + "-" + parts[-1])[0]

def is_mp3(path):
    return os.path.splitext(path)[1] == ".mp3" or os.path.splitext(path)[1] == ".wav"

def get_voicery_transcript(path):
    base = os.path.split_ext(path)[0]

    normalized_path = base + ".aligned.txt"

    with open(normalized_path) as normalized_transcript_file:
        return normalized_transcript_file.read().strip()

def get_cc_search_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-west-europe/archive_org/v1/data.csv")

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



