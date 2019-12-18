# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from data_generator.mnist_generator import DataGenerator
from models.mnist_model import MNISTModel
from trainers.mnist_trainer import MNISTTrainer
from utils.logger import Logger
from utils.utils import *


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        print()

    except Exception as e:
        print("Exception: {}".format(e))
        print("missing or invalid arguments in config json file: {}".format(args.config))
        exit(0)

    # create the experiments dirs
    create_dirs([config["summary_dir"], config["checkpoint_dir"], config["log_dir"], config["model_dir"]])
    
    # create tensorflow session
    sess = tf.Session()

    # create tensorboard logger
    logger = Logger(sess, config)
   
    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you want
    model = MNISTModel(config, logger)
    
 
    
    # create trainer and pass all the previous components to it
    trainer = MNISTTrainer(sess, model, data, config, logger)
    
    #load model if exists
    if config["transfer_learning"]:
        model.load_checkpoint(sess)
    
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
