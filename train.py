
import sys
import os
import json

from argparse import ArgumentParser

def main():

    parser = ArgumentParser(description="A script for training and inference using "
        "the The People's Speech.")

    parser.add_argument("-n", "--experiment-name", default = "models/peoples-speech",
        help = "A unique name for the experiment.")
    parser.add_argument("-p", "--predict", default = False, action="store_true",
        help = "Run prediction on a specified trained model instead of training.")
    parser.add_argument("-m", "--model-path", default = "configs/sota.json",
        help = "Load the specified model or config file.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-L", "--enable-logger", default = [], action="append",
        help = "Enable logging for a specific module")
    parser.add_argument("-O", "--override-config", default = [], action="append",
        help = "Override config file arguments")
    parser.add_argument("--run-in-this-repo", default = False, action="store_true",
        help = "Run code from this repo rather than from the repo with the model (if different).")
    parser.add_argument("--test-set", default="",
        help = "The path to the test set to run on.")
    parser.add_argument("--test-set-type", default="AudioCsvDataset",
        help = "The type of dataset.")
    parser.add_argument("-o", "--output-path", default="predictions.json",
        help = "The output path the save the output of the inference run.")

    arguments = vars(parser.parse_args())

    run_locally(arguments)

def run_locally(arguments):
    with open(arguments["model_path"]) as config_file:
        config = json.load(config_file)

    if "directory" in config and not arguments["run_in_this_repo"]:
        directory = config["directory"]
    else:
        directory = os.path.dirname(os.path.abspath(__file__))

    sys.path.append(os.path.join(directory, 'source'))

    import engine.run_locally

    engine.run_locally.run_locally(arguments)


################################################################################
## Guard Main
if __name__ == "__main__":
    main()
################################################################################



