import tensorflow as tf


class BaseTrainer:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):

        if self.config["transfer_learning"] is True:
            self.sess.run(tf.assign(self.model.current_epoch_tensor, 0))
            print("transfer learning mode, assign epoch = 0.")
        else:
            print("regular training mode, current epoch: {}.".format(self.sess.run(self.model.current_epoch_tensor)))
        
        for current_epoch in range(self.model.current_epoch_tensor.eval(self.sess), self.config["num_epochs"] + 1, 1):
            self.train_epoch(current_epoch)
            self.sess.run(self.model.increment_current_epoch_tensor)

    def train_epoch(self, current_epoch):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self, current_epoch, current_iter):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
