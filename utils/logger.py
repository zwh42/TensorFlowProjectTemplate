import tensorflow as tf
import os
import logging


class Logger:
    def __init__(self, config):
        
        self.config = config
        
        self._logger = {}
        self._logger["flow"] = self.add_logger("flow", log_file = os.path.join(self.config["log_dir"], "flow.log"), \
            message_format = '[%(asctime)s] %(message)s', level=logging.INFO, print_to_screen = True)
        
        self._logger["train"] = self.add_logger("train", log_file = os.path.join(self.config["log_dir"], "training_record.log"), \
            message_format = '%(message)s', level=logging.INFO, print_to_screen = True)

 
    def add_logger(self, name, log_file, message_format = '[%(asctime)s] %(message)s', level=logging.INFO, print_to_screen = True):
        """ set up logger """
        date_format = "%Y-%m-%d_%H:%M:%S"
        formatter = logging.Formatter(message_format, date_format)

        handler = logging.FileHandler(log_file, mode = "w")        
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        if print_to_screen:
            logger.addHandler(logging.StreamHandler())

        return logger
    

    def logging(self, name, message):
        if name not in self._logger:
            raise Exception("logger name {} does not exists!".format(name))
            
        self._logger[name].info(message)