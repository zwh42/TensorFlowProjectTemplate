from base.base_trainer import BaseTrainer
import numpy as np


class MNISTTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config,logger):
        super(MNISTTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self, current_epoch):
        loop = range(self.config["num_iter_per_epoch"])
        losses = []
        accs = []
        for current_iter in loop:
            loss, acc = self.train_step(current_epoch, current_iter)
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self, current_epoch, current_iter):
        batch_x, batch_y = next(self.data.next_batch(self.config["batch_size"]))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.acc],
                                     feed_dict=feed_dict)
        return loss, acc
