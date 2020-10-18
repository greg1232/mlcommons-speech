
from google.cloud import storage
from argparse import ArgumentParser
import logging
import csv
import os
import json
from pydub import AudioSegment

logger = logging.getLogger(__name__)

from smart_open import open

def make_splits(arguments):
    samples = []

    #get_common_voice_samples(samples)
    #get_librispeech_samples(samples)
    #get_librivox_samples(samples)
    #get_voicery_samples(samples)
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
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librivox-v0.1/data.csv")

def get_voicery_samples(samples):
    mp3_files = get_mp3_files("gs://the-peoples-speech-aws-import/voicery")

    for path, name in mp3_files.items():
        transcript = get_voicery_transcript(path)

        samples.append((path, transcript, {"speaker_id" : "voicery_" + name}))

def get_voicery_transcript(path):
    base = os.path.split_ext(path)[0]

    normalized_path = base + ".aligned.txt"

    with open(normalized_path) as normalized_transcript_file:
        return normalized_transcript_file.read().strip()

def get_cc_search_samples(samples):
    extract_aligned_samples(samples, "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020",
        "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020_aligned_data_9_15_20")

def extract_aligned_samples(samples, audio_path, alignment_path):
    mp3_files = get_mp3_files(audio_path)

    storage_client = storage.Client()

    bucket_name, prefix = get_bucket_and_prefix(audio_path)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    for blob in blobs:
        if is_aligned_file(blob.name):
            alignment_file_name = "gs://" + os.path.join(bucket_name, blob.name)
            alignments, mp3_path = load_alignments(alignment_file_name)

            if not blob_exists(mp3_files, mp3_path):
                continue

            mp3_size = get_blob_size(mp3_path)

            if mp3_size > 75e6:
                logger.debug("Skipping mp3 from " + mp3_path + " with " + str(mp3_size / 1e6) + "MB which is too big")
                continue

            logger.debug("Loading mp3 from " + mp3_path + " with " + str(mp3_size / 1e6) + "MB")
            mp3 = load_audio(mp3_path)

            for index, alignment in enumerate(alignments):
                start_time = alignment["start"]
                end_time = alignment["end"]
                transcript = alignment["aligned"]
                metadata = alignment
                path = os.path.splitext(mp3_path)[0] + "-" + str(index) + ".mp3"

                aligned_path = make_alignment(mp3_files, mp3, path, start_time, end_time)

                samples.append({"path" : aligned_path, "caption" : transcript, "metadata" : metadata})

            # Optionally destroy the object and force the garbage collector to run
            import gc
            del mp3
            gc.collect()

storage_client = storage.Client()

def get_blob_size(path):

    bucket_name, prefix = get_bucket_and_prefix(path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(prefix)
    return blob.size

def blob_exists(paths, path):
    return path in paths

class MP3File:
    def __init__(self, mp3_path):
        self.mp3_path = mp3_path
        self.mp3 = None

    def get(self):
        if self.mp3 is None:
            with open(self.mp3_path, 'rb', buffering=0) as mp3_file:
                self.mp3 = AudioSegment.from_mp3(mp3_file)

        return self.mp3

def load_audio(mp3_path):
    return MP3File(mp3_path)

def make_alignment(mp3_files, mp3, path, start_time, end_time):

    if blob_exists(mp3_files, path):
        logger.debug("Skipping existing alignment " + path)
    else:
        segment = mp3.get()[start_time:end_time]
        logger.debug("Saving alignment to " + path)
        with open(path, "wb", buffering=0) as mp3_file:
            segment.export(mp3_file, format="mp3")
        del segment

def is_aligned_file(path):
    return path.find("aligned.json") != -1

def load_alignments(path):
    logger.debug("Loading " + path)
    with open(path) as alignment_file:
        return json.load(alignment_file), get_mp3_path_for_aligned_file(path)

def get_mp3_path_for_aligned_file(path):
    # gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020_aligned_data_9_15_20/CAPTIONED_DATA/output/10_10_2017_Essex_Junction_Trustees/aligned.json
    # gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020/CAPTIONED_DATA/10_10_2017_Essex_Junction_Trustees/10_10_2017_Essex_Junction_Trustees.mp3

    parts = split_all(path[5:])

    return "gs://" + os.path.join(*(parts[:2] + ["Aug_18_2020", "CAPTIONED_DATA"] + parts[-2:-1] + parts[-2:-1])) + ".mp3"

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

def get_mp3_files(audio_path):
    storage_client = storage.Client()

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


def get_bucket_and_prefix(path):
    parts = split_all(path[5:])

    return parts[0], os.path.join(*parts[1:])

def is_mp3(path):
    return os.path.splitext(path)[1] == ".mp3" or os.path.splitext(path)[1] == ".wav"

def get_key(path):
    parts = split_all(path)
    return os.path.splitext(parts[-2] + "-" + parts[-1])[0]

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



