import tensorflow as tf
from tensorflow.python.framework import graph_util

class BaseModel:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger.logger
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_current_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save_checkpoint(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def save_to_protobuf(self, sess, output_node_node, model_path):
        var_list = tf.global_variables()
        constant_graph = constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [output_node_node])

        with tf.gfile.FastGFile(model_path,  mode = 'wb') as f: 
            f.write(constant_graph.SerializeToString())
        print("model:{} in protobuf format is saved.".format(model_path))


    # load latest checkpoint from the experiment path defined in the config file
    def load_checkpoint(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config["checkpoint_dir"])
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("model checkpoint loaded.")
        else:
            print("loading checkpoint failed.")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_current_epoch(self):
        with tf.variable_scope('current_epoch'):
            self.current_epoch_tensor = tf.Variable(0, trainable=False, name='current_epoch')
            self.increment_current_epoch_tensor = tf.assign(self.current_epoch_tensor, self.current_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
