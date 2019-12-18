import tensorflow as tf
import os
import logging


class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "train"),
                                                          self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "test"))

        self.logger = {}
        
        self.logger["flow"] = self.setup_logger("flow", log_file = os.path.join(self.config["log_dir"], "flow.log"), \
            format = '[%(asctime)s] %(message)s', level=logging.INFO, print_to_screen = True)
        
        self.logger["train"] = self.setup_logger("train", log_file = os.path.join(self.config["log_dir"], "training_record.log"), \
            format = '%(message)s', level=logging.INFO, print_to_screen = True)

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()


    def setup_logger(self, name, log_file, format = '[%(asctime)s] %(message)s', level=logging.INFO, print_to_screen = True):
        """ set up logger """
        formatter = logging.Formatter(format)

        handler = logging.FileHandler(log_file, mode = "w")        
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        if print_to_screen:
            logger.addHandler(logging.StreamHandler())

        return logger