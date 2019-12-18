import numpy as np
from tensorflow.examples.tutorials.mnist import input_data



class DataGenerator:
    def __init__(self, config):

        self.config = config
        
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        '''
        self.x = np.load(X)
        self.y = np.load(Y)

        idx = np.arange(0 , len(self.x))
        np.random.shuffle(idx)
        self.x = [self.x[i] for i in idx]
        self.y = [self.y[i] for i in idx]

        self.x = np.array(self.x)
        self.y = np.array(self.y) 
        print("training sample shape: X = {}, Y = {}".format(self.x.shape, self.y.shape))
        if len(self.x) != len(self.y):
            print("Error! Training sample size is not equal.")
        '''
    

    def next_batch(self, batch_size):
        x, y = self.mnist.train.next_batch(batch_size)
        x = np.reshape(x, (batch_size, 28, 28))
        #y = np.reshape(y, (batch_size, 10, 1))
        x = np.asarray(x)
        y = np.asarray(y)
        
        #print("shape", x.shape, y.shape)


        yield np.asarray(x), np.asarray(y)
    

