from argparse import ArgumentParser
import logging
import csv
import json
import os
import queue
import threading
import gc

from google.cloud import storage

from pydub import AudioSegment

logger = logging.getLogger(__name__)

storage_client = storage.Client()

def main():
    parser = ArgumentParser("This program converts the DSAlign "
        "data format to the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020",
        help = "The GCP path to the audio dataset.")
    parser.add_argument("--aligned-path", default = "gs://the-peoples-speech-west-europe/archive_org/Aug_18_2020_aligned_data_9_15_20",
        help = "The GCP path to the alignment dataset.")
    parser.add_argument("--max-count", default = 1e9,
        help = "The maximum number of audio samples to extract.")
    parser.add_argument("--cache-directory", default = "",
        help = "The local path to cache.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "The output path to save the dataset.")
    parser.add_argument("--worker-count", default = 4,
        help = "The number of worker threads.")
    parser.add_argument("-v", "--verbose", default = False,
        action="store_true",
        help = "Set the log level to debug, "
            "printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    arguments["system"] = {"cache-directory" : arguments["cache_directory"]}

    setup_logger(arguments)

    convert_cc_search_to_csv(arguments)

def convert_cc_search_to_csv(arguments):

    directory = arguments["output_path"]

    logger.debug("Checking directory: " + directory)
    if not os.path.exists(directory):
        logger.debug("Making directory: " + directory)
        os.makedirs(directory)

    with open(os.path.join(arguments["output_path"], "data.csv"), "w", newline="") as output_csv_file, \
        open(os.path.join(arguments["output_path"], "metadata.csv"), "w", newline="") as metadata_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')
        metadata_writer = csv.writer(metadata_csv_file, delimiter=',', quotechar='"')
        update_csv(arguments, csv_writer, metadata_writer)

def update_csv(arguments, csv_writer, metadata_writer):
    mp3_files = get_mp3_files(arguments["input_path"])

    total_count = 0

    file_uploader = FileUploader(arguments)

    for bucket_name, file_name in get_all_object_paths(arguments):
        if not is_aligned_file(file_name):
            continue

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

        logger.debug("Extracting alignments from " + str(alignment_file_name) + ", " + str(mp3_path))

        for entry in alignments:
            start = entry["start"]
            end = entry["end"]

            text = entry["aligned"]

            save_training_sample(mp3_files, file_uploader, csv_writer, metadata_writer, audio, start, end, text, entry, arguments, total_count)

            total_count += 1

            if total_count >= int(arguments["max_count"]):
                break

        del alignments

        delete_audio(mp3, bucket_name, file_name, arguments)

        if total_count >= int(arguments["max_count"]):
            break

    file_uploader.join()

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

def is_mp3(path):
    return os.path.splitext(path)[1] == ".mp3" or os.path.splitext(path)[1] == ".wav"

def blob_exists(paths, path):
    return path in paths

def get_blob_size(path):
    bucket_name, prefix = get_bucket_and_prefix(path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(prefix)
    return blob.size

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

def load_audio(bucket_name, path, arguments):
    return MP3File(bucket_name, path, arguments)

class MP3File:
    def __init__(self, bucket_name, path, arguments):
        self.bucket_name = bucket_name
        self.path = path
        self.arguments = arguments
        self.mp3 = None

    def get(self):
        if self.mp3 is None:
            local_cache = LocalFileCache(self.arguments, "gs://" + os.path.join(self.bucket_name, self.path)).get_path()

            self.mp3 = AudioSegment.from_mp3(local_cache)

        return self.mp3

def delete_audio(mp3, bucket_name, path, arguments):
    del mp3
    gc.collect()
    local_cache = LocalFileCache(arguments, "gs://" + os.path.join(bucket_name, path)).get_path()
    if os.path.exists(local_cache):
        os.remove(local_cache)

def extract_audio(audio, start, end):
    return audio[start:end].get()

def save_training_sample(mp3s, file_uploader, csv_writer, metadata_writer, audio, start, end, text, entry, arguments, total_count):
    path = get_output_path(arguments, total_count)

    if not blob_exists(mp3s, path):
        local_path = get_local_path(arguments, total_count)

        directory = os.path.dirname(local_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        logger.debug("Saving sample: " + path + " at local path " + local_path)

        audio_segment = extract_audio(audio, start, end)

        audio_segment.export(local_path, format="wav")

        file_uploader.upload(path, local_path)

    csv_writer.writerow([path, text])
    metadata_writer.writerow([path, json.dumps(entry)])

def get_output_path(arguments, total_count):
    return os.path.join(arguments["output_path"], "data", str(total_count) + ".wav")

def get_local_path(arguments, total_count):
    bucket, key = get_bucket_and_prefix(get_output_path(arguments, total_count))
    return os.path.join(arguments["system"]["cache-directory"], key)

def get_all_object_paths(arguments):

    bucket_name, prefix = get_bucket_and_prefix(arguments["aligned_path"])
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    for blob in blobs:
        yield bucket_name, blob.name

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

def setup_logger(arguments):

    if arguments["verbose"]:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s -'
        ' %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

class FileUploader:
    def __init__(self, arguments):
        self.queue = queue.Queue(maxsize=512)
        self.threads = []

        for i in range(int(arguments["worker_count"])):
            thread = threading.Thread(target=upload_files_worker, args=(self.queue,))
            thread.start()
            self.threads.append(thread)

    def upload(self, path, local_path):
        self.queue.put((path, local_path, False))

    def join(self):

        for thread in self.threads:
            self.queue.put((None, None, True))

        for thread in self.threads:
            thread.join()

def upload_files_worker(queue):
    while True:
        path, local_path, is_finished = queue.get()

        if is_finished:
            break

        logger.debug("Uploading " + local_path + " to " + path)

        bucket_name, key = get_bucket_and_prefix(path)
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(key)

        blob.upload_from_filename(local_path)

        os.remove(local_path)

class LocalFileCache:
    """ Supports caching.  Currently it supports read-only access to GCS.
    """

    def __init__(self, config, remote_path):
        self.config = config
        self.remote_path = remote_path

        self.download_if_remote()

    def get_path(self):
        return self.local_path

    def download_if_remote(self):
        if not self.is_remote_path(self.remote_path):
            self.local_path = self.remote_path
            return

        self.local_path = self.compute_local_path()

        self.download()

    def download(self):
        if os.path.exists(self.get_path()):
            logger.info(" using cached file '" + self.get_path() + "'")
            return

        directory = os.path.dirname(self.get_path())

        os.makedirs(directory, exist_ok=True)

        bucket, key = get_bucket_and_prefix(self.remote_path)

        logger.info(
            "Downloading '" + self.remote_path + "' to '" + self.get_path() + "'"
        )

        bucket = storage_client.get_bucket(bucket)
        blob = bucket.get_blob(key)

        blob.download_to_filename(self.local_path)

    def is_remote_path(self, path):
        return path.find("gs:") == 0

    def compute_local_path(self):
        bucket, key = get_bucket_and_prefix(self.remote_path)
        return os.path.join(self.config["system"]["cache-directory"], key)




main()












