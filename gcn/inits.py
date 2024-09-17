import tensorflow as tf
import numpy as np
# tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.compat.v1.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None, seed=42):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    tf.random.set_seed(seed)
    initial = tf.compat.v1.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float64, seed=seed)
    return tf.Variable(initial, name=name)


def glorot_dec(shape, name=None, seed=42):
    """Glorot & Bengio (AISTATS 2010) init."""
    num_nodes, feat_in, feat_out = shape
    init_range = np.sqrt(6.0/(shape[1]+shape[2]))
    tf.random.set_seed(seed)
    initial = tf.compat.v1.random_uniform((feat_in, feat_out), minval=-init_range, maxval=init_range, dtype=tf.float64, seed=seed)
    initial = tf.tile(initial[tf.newaxis], [num_nodes, 1, 1])
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
