
from bs4 import BeautifulSoup

from warcio.archiveiterator import ArchiveIterator

import csv
import cld3

from argparse import ArgumentParser

def main():

    parser = ArgumentParser(description="Convert WARC archive to csv, one line per entry.")


    parser.add_argument("-o", "--output-path", default="",
        help = "The input path to read a warc from.")

    parser.add_argument("-c", "--count", default=1e4,
        help = "How many lines to extract.")

    parser.add_argument("-i", "--input-path", default="",
        help = "The output path the save csv to.")

    arguments = vars(parser.parse_args())

    convert_warc_to_csv(arguments)

def convert_warc_to_csv(arguments):
    counter = 0
    with open(arguments["input_path"], 'rb') as input_file, \
         open(arguments["output_path"], "w") as output_file:

        writer = csv.writer(output_file, delimiter=',', quotechar='"')

        for record in ArchiveIterator(input_file):

            if record.rec_type == 'response':
                if record.http_headers.get_header('Content-Type') == 'text/html':
                    html = record.content_stream().read()
                    clean_lines = clean_html(html)
                    for line in clean_lines:
                        language_prediction = cld3.get_language(line)
                        if language_prediction.language == 'en':
                            writer.writerow([line, language_prediction])
                            counter += 1

                            if counter >= int(arguments["count"]):
                                return



def clean_html(html):
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    return [chunk for chunk in chunks if chunk]


if __name__ == "__main__":
    main()



