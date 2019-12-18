import tensorflow as tf
import numpy as np

# Create some layer wrappers for simplicity

def conv2d(x, W, b, strides=1, padding = "VALID",  name=None):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = padding, name = name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, padding = "VALID"):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding= padding)

def batch_normalization(x, is_training=True, scale=True, updates_collections=None):
    return tf.contrib.layers.batch_norm(x, is_training=is_training, scale=scale, updates_collections=updates_collections)

def weight_variable(shape, stddev = 0.02, name = None):
    initial = tf.truncated_normal(shape, stddev = stddev)
    #initial = tf.contrib.layers.xavier_initializer(uniform=False)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer = initial)

def bias_variable(shape, name = None):
    initial = tf.constant(0.0, shape = shape)
    #initial = tf.contrib.layers.xavier_initializer(uniform=False)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer = initial)

def gaussian(size, sigma = 1.0):
    x, y = np.mgrid[-size:size + 1, -size: size + 1]
    g = np.exp(-(x**2+y**2)/float(sigma))
    return g / g.sum()

def smooth(layer, name = None):
    kernel = gaussian(2)
    kernel = tf.cast(tf.reshape(kernel, [5, 5, 1, 1]), tf.float32)
    return tf.nn.conv2d(layer, kernel, [1, 1, 1, 1], padding='VALID', name = name)