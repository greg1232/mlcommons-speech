
import numpy
import logging
import json
import os
import inspect
import shutil

from model.ModelFactory import ModelFactory
from data.DataSources import DataSources
from data.DataSourceFactory import DataSourceFactory
from inference.PredictorFactory import PredictorFactory

from tensorflow.python.client import device_lib
import tensorflow as tf

def run_locally(arguments):

    setup_logging(arguments)

    config = load_config(arguments)

    device = setup_device(config)

    with tf.device(device):
        if arguments["predict"]:
            test_data = get_data(config, "test-data-sources")
            predictor = get_predictor(config, test_data)

            predictor.predict()

        else:
            make_experiment(config, arguments)

            training_data   = get_data(config, "training-data-sources")
            validation_data = get_data(config, "validation-data-sources")

            model = get_model(config, training_data, validation_data)
            model.train()

def setup_logging(arguments):
    numpy.set_printoptions(precision=3, linewidth=150)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    root_logger.addHandler(ch)

    for scope in arguments["enable_logger"]:
        logger = logging.getLogger(scope)
        logger.setLevel(logging.DEBUG)

def get_model(config, training_data, validation_data):
    return ModelFactory(config,
        training_data,
        validation_data).create()

def get_predictor(config, validation_data):
    return PredictorFactory(config, validation_data).create()

def get_data(config, name):
    sources = config[name]

    data_sources = DataSources(config)

    for source in sources:
        data_sources.add_source(DataSourceFactory(config).create(source))

    return data_sources

def make_experiment(config, arguments):
    config["model"]["directory"] = name_directory(arguments["experiment_name"])

    directory = config["model"]["directory"]

    if not os.path.exists(directory):
        os.makedirs(directory)

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
        config["test-data-sources"] = [{ "type" : arguments["data_source_type"],
                                         "path" : arguments["test_set"] }]

    override_config(config, arguments)

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

