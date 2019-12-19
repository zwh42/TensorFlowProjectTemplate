from base.base_model import BaseModel
import tensorflow as tf
from wrappers.layer_wrappers import *



class MNISTModel(BaseModel):
    def __init__(self, config, logger):
        super(MNISTModel, self).__init__(config, logger)
             
        self._input_node_name = self.config["input_node_name"]
        self._output_node_name = self.config["output_node_name"]

        self.logger.logging("flow", "model input node name: {}".format(self._input_node_name))
        self.logger.logging("flow", "model output node name: {}".format(self._output_node_name))
        
        self.build_model()
        self.init_saver()

    
    @property
    def input_node_name(self):
        return self._input_node_name
    
    @property
    def output_node_name(self):
        return self._output_node_name
    
    @property
    def loss(self):
        return self._loss

    @property
    def acc(self):
        return self._acc
    
    def build_model(self):
        self.logger.logging("flow", "build model: ")
        self.is_training = tf.placeholder(tf.bool)
        ## define input output node, name is needed for protobuf saving
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config["input_size"], name = self._input_node_name)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config["output_size"], name = self._output_node_name)

       
        # network architecture
        layer = tf.reshape(self.x, [-1, 28, 28, 1])
        layer = conv2d(layer, weight_variable([5, 5, 1, 32]), bias_variable([32]), padding='SAME')
        layer = maxpool2d(layer, k = 2)
        layer = conv2d(layer, weight_variable([5, 5, 32, 64]), bias_variable([64]), padding='SAME')
        layer = maxpool2d(layer, k = 2)
        
        self.logger.logging("flow", "before flatten shape: {}".format(layer.shape))
        layer = tf.reshape(layer, [-1, 7*7*64])
        self.logger.logging("flow", "after flatten shape: {}".format(layer.shape))
        layer = tf.layers.dense(layer, 32)
        layer = tf.layers.dense(layer, 10)    
        self.model_output = tf.identity(layer, name = self._output_node_name)        

        self.set_train_step(self.y, self.model_output)


    def set_train_step(self, y_true, y_pred):
        
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_pred, labels = y_true))
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        self._acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self.train_step = tf.train.AdamOptimizer(self.config["learning_rate"])
            self.train_step = self.optimizer.minimize(self._loss, global_step = self.global_step_tensor)


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

