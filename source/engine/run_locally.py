
import logging
import json
import os
import inspect
import shutil

from model.AcousticModelFactory import AcousticModelFactory
from model.LanguageModelFactory import LanguageModelFactory
from dataset.DatasetFactory import DatasetFactory
from inference.PredictorFactory import PredictorFactory

from tensorflow.python.client import device_lib
import tensorflow as tf

def run_locally(arguments):

    setup_logging(arguments)

    config = load_config(arguments)

    device = setup_device(config)

    with tf.device(device):
        if arguments["predict"]:
            run_predict(config)
        else:
            make_experiment(config, arguments)
            run_training(config)

def run_predict(config):
    test_data = get_data(config, "test-set")
    predictor = PredictorFactory(config, test_data).create()

    predictor.predict()

def run_training(config):
    run_language_model_training(config)
    run_acoustic_model_training(config)

def run_language_model_training(config):
    training_data = get_data(config, "language-training-set")
    development_data = get_data(config, "language-development-set")

    model = LanguageModelFactory(config, training_data, development_data).create()

    model.train()

def run_acoustic_model_training(config):
    training_data = get_data(config, "acoustic-training-set")
    development_data = get_data(config, "acoustic-development-set")

    model = AcousticModelFactory(config, training_data, development_data).create()

    model.train()

def get_data(config, name):
    return DatasetFactory(config).create(config[name])

def setup_logging(arguments):

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

def make_experiment(config, arguments):
    directory = name_directory(arguments["experiment_name"])

    config["directory"] = directory

    if not "directory" in config["acoustic-model"]:
        config["acoustic-model"]["directory"] = os.path.join(directory, "acoustic-model")

    if not "directory" in config["language-model"]:
        config["language-model"]["directory"] = os.path.join(directory, "language-model")

    print("lm  directory", config["language-model"]["directory"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(config["acoustic-model"]["directory"]):
        os.makedirs(config["acoustic-model"]["directory"])

    if not os.path.exists(config["language-model"]["directory"]):
        os.makedirs(config["language-model"]["directory"])

    # save the config file
    config_path = os.path.join(directory, "config.json")

    with open(config_path, 'w') as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)

    # save the code
    code_directory = os.path.join(directory, "source")
    code_file = os.path.join(directory, "train.py")

    current_file_path  = os.path.abspath(inspect.getfile(inspect.currentframe()))
    current_code_path  = os.path.dirname(os.path.dirname(current_file_path))
    training_file_path = os.path.join(os.path.dirname(current_code_path), "train.py")

    shutil.copytree(current_code_path, code_directory, ignore=shutil.ignore_patterns(("^.py")))
    shutil.copyfile(training_file_path, code_file)

def name_directory(directory):
    extension = 0

    directory = os.path.abspath(directory)

    while os.path.exists(directory + '-' + str(extension)):
        extension += 1

    return directory + '-' + str(extension)

def override_config(config, arguments):
    for override in arguments["override_config"]:
        path, value = override.split('=')
        components = path.split('.')

        local_config = config
        for i, component in enumerate(components):
            if i == len(components) - 1:
                local_config[component] = value
            else:
                if not component in local_config:
                    local_config[component] = {}
                local_config = local_config[component]

def load_config(arguments):

    with open(arguments["model_path"]) as config_file:
        config = json.load(config_file)

    if len(arguments["test_set"]) > 0:
        config["test-set"] = { "path" : arguments["test_set"], "type" : arguments["test_set_type"] }

    override_config(config, arguments)

    config["output-path"] = arguments["output_path"]

    return config

def setup_device(config):
    if has_gpus():
        return "/device:GPU:0"
    else:
        return "/cpu:0"

def has_gpus():
    return len(get_available_gpus()) > 0

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

