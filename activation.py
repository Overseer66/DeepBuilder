import tensorflow as tf
from .util import *


def Activation(x, activation, name=''):
    return activation(x, name=name)


def LeakyReLU(x, leak=0.2, name='LeakyReLU'):
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def LinearUnit(x, name='Linear'):
    with tf.name_scope(name):
        return x


def Scale(x, width, height, params=(), name=''):
    return tf.image.resize_nearest_neighbor(x, (width, height), **params, name=name)


def BatchNorm(x, params=(), name=''):
    return tf.layers.batch_normalization(x, **params, name=name)


def Softmax(
        input,
        name='_Softmax',
        layer_collector=None,
):
    l = tf.nn.softmax(input, name=name)
    safe_append(layer_collector, l)

    return l