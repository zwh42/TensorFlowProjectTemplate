import numpy as np


class DataGenerator:
    def __init__(self, X, Y):

        max_count = 2500
        
        self.x = np.load(X)
        self.y = np.load(Y)

        idx = np.arange(0 , len(self.x))
        np.random.shuffle(idx)
        self.x = [self.x[i] for i in idx]
        self.y = [self.y[i] for i in idx]

        self.x = np.array(self.x[:max_count])
        self.y = np.array(self.y[:max_count]) 
        print("training sample shape: X = {}, Y = {}".format(self.x.shape, self.y.shape))
        if len(self.x) != len(self.y):
            print("Error! Training sample size is not equal.")
    

    def next_batch(self, batch_size):
        idx = np.arange(0 , len(self.x))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        x = [self.x[i] for i in idx]
        y = [self.y[i] for i in idx]
        
        yield np.asarray(x), np.asarray(y)
    
    def all(self):
        return np.asarray(self.x), np.asarray(self.y)
