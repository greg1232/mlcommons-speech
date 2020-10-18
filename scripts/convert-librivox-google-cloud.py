from argparse import ArgumentParser
import logging
import csv
import json
import os
import queue
import threading
from google.cloud import storage

from pydub import AudioSegment

logger = logging.getLogger(__name__)

storage_client = storage.Client()

def main():
    parser = ArgumentParser("This program converts the DSAlign "
        "data format to the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The GCP path to the DSAligned dataset.")
    parser.add_argument("--max-count", default = 1e9,
        help = "The maximum number of audio samples to extract.")
    parser.add_argument("--cache-directory", default = "data",
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

    convert_dsalign_to_csv(arguments)


def convert_dsalign_to_csv(arguments):

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

    total_count = 0

    file_uploader = FileUploader()

    for bucket_name, file_name in get_all_object_paths(arguments):
        if not is_audio(file_name):
            continue

        aligned_file_name = get_corresponding_align_file_name(file_name)

        if not exists(bucket_name, aligned_file_name):
            continue

        logger.debug("Extracting alignments from " + str(aligned_file_name) + ", " + str(file_name))

        alignment = load_alignment(bucket_name, aligned_file_name, arguments)

        audio = load_audio(bucket_name, file_name, arguments)

        for entry in alignment:
            start = entry["start"]
            end = entry["end"]

            text = entry["aligned"]

            audio_segment = extract_audio(audio, start, end)

            save_training_sample(file_uploader, csv_writer, metadata_writer, audio_segment, text, entry, arguments, total_count)

            total_count += 1

            if total_count >= int(arguments["max_count"]):
                return

        delete_audio(bucket_name, file_name, arguments)

def is_audio(path):
    return os.path.splitext(path)[1] == '.mp3'

def get_corresponding_align_file_name(path):
    return os.path.splitext(path)[0] + ".aligned"

def exists(bucket_name, prefix):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(prefix)
    if blob is None:
        return False
    return blob.exists()

def load_alignment(bucket_name, path, arguments):
    local_cache = LocalFileCache(arguments, "gs://" + os.path.join(bucket_name, path)).get_path()
    with open(local_cache) as json_file:
        return json.load(json_file)

def load_audio(bucket_name, path, arguments):
    local_cache = LocalFileCache(arguments, "gs://" + os.path.join(bucket_name, path)).get_path()

    return AudioSegment.from_mp3(local_cache)

def delete_audio(bucket_name, path, arguments):
    local_cache = LocalFileCache(arguments, "gs://" + os.path.join(bucket_name, path)).get_path()
    os.remove(local_cache)

def extract_audio(audio, start, end):
    return audio[start:end]

def save_training_sample(file_uploader, csv_writer, metadata_writer, audio_segment, text, entry, arguments, total_count):
    path = get_output_path(arguments, total_count)
    local_path = get_local_path(arguments, total_count)

    directory = os.path.dirname(local_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    logger.debug("Saving sample: " + path + " at local path " + local_path)

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

    bucket_name, prefix = get_bucket_and_prefix(arguments["input_path"])
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

        for i in int(argumnts["worker_count"]):
            thread = threading.Thread(target=upload_files_worker, args=(self.queue,))
            thread.start()

    def upload(self, path, local_path):
        self.queue.put((path, local_path))


def upload_files_worker(queue):
    while True:
        path, local_path = queue.get()

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













