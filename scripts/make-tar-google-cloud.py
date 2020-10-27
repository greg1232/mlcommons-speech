
from google.cloud import storage
from argparse import ArgumentParser
import logging
import csv
import os
import json
import tarfile
from smart_open import open

logger = logging.getLogger(__name__)

storage_client = storage.Client()

def main():
    parser = ArgumentParser("Takes all of the files in a dataset and uploads into a TAR.GZ archive .")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.5/train.csv",
        help = "The output path to load the dataset from.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.5/train.tar.gz",
        help = "The output path to save the test dataset.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    make_tar_gz(arguments)

def make_tar_gz(arguments):
    samples = load_csv(arguments["input_path"])

    archive = open_archive(arguments["output_path"])

    updated_samples = [ (update_path(sample["path"]), sample["path"], sample["caption"], sample["metadata"]) for sample in samples]

    writer = ArchiveWriter(archive, updated_samples)

    writer.run()

def update_path(path):
    bucket_name, prefix = get_bucket_and_prefix(path)
    return prefix

def get_bucket_and_prefix(path):
    parts = split_all(path[5:])

    assert len(parts) > 1, str(parts)

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

def load_csv(csv_path):
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

    return new_samples


def open_archive(path):
    tar_file = open(path, mode="wb")
    return tarfile.TarFile(fileobj=tar_file, mode="w"), tar_file

class ArchiveWriter:
    def __init__(self, archive_path, samples):
        self.archive, self.archive_file = load_archive(archive_path)
        self.samples = samples

        self.csv_file_name = "data.csv"
        self.csv_file = open(self.csv_file_name, newline="", mode="w")
        self.csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"')

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            future_to_data = {executor.submit(load_file, path): (updated_path, path, transcript, metadata) for updated_path, path, transcript, metadata in self.samples}
            for future in concurrent.futures.as_completed(future_to_data):
                updated_path, path, transcript, metadata = future_to_data[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (path, exc))
                else:
                    logger.debug("loaded %s bytes from %s " % (len(data), path))

                self.archive.addfile(updated_path, data)
                self.csv_writer.writerow([updated_path, transcript, metadata])

        self.csv_file.close()

        self.archive.add(self.csv_file_name)
        self.archive.close()
        self.archive_file.close()


def load_file(path):
    bucket_name, prefix = get_bucket_and_prefix(normalized_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(prefix)

    data = blob.download_as_bytes()

    return io.BytesIO(data)


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

