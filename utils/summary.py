import tensorflow as tf
import os



class Summary:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.summary_writer = {}
        self.summary_writer["train"] = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "train"),
                                                          self.sess.graph)
        self.summary_writer["test"] = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "test"))

        #self.summary_writer["train"].add_graph(self.sess.graph)


    # it can summarize scalars and images.
    def summarize(self, step, summarizer = "train", scope = "", summaries_dict = None):
        
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        
        if summarizer == "train":
            summary_writer = self.summary_writer["train"]
        elif summarizer == "test":
            summary_writer = self.summary_writer["test"]

        
        with tf.variable_scope(scope):
            print("write summary:")
            print(summaries_dict)
            
            summary_writer.add_graph(self.sess.graph)

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