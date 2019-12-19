import argparse
import json
import os
from pprint import pprint


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        required=True,
        help='The configuration json file')
    args = argparser.parse_args()
    
    return args




def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
       
        pprint(config_dict)

    return  config_dict


def process_config(json_file):
    config = get_config_from_json(json_file)
    
    ##add a few dir for later use
    config["summary_dir"] = os.path.join("./experiments", config["exp_name"], "summary/")
    config["checkpoint_dir"] = os.path.join("./experiments", config["exp_name"], "checkpoint/")
    config["log_dir"] = os.path.join("./experiments", config["exp_name"], "logs/")
    config["model_dir"] = os.path.join("./experiments", config["exp_name"], "model/")    
    
    return config



def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        return 0
    
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
