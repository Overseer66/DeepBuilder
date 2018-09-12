import tensorflow as tf
from .util import *


def Activation(input, activation, kwargs={}, name=None, layer_collector=None):
    name = activation.name if name = None else name
    l = activation(input, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def LeakyReLU(input, leak=0.2, name='_LeakyReLU', layer_collector=None):
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        l = f1 * input + f2 * abs(input)
        safe_append(l)
        return l


def Scale(input, width, height, kwargs={}, name='_Scale', layer_collector=None):
    l = tf.image.resize_nearest_neighbor(input, (width, height), **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def BatchNorm(input, kwargs={}, name='_BatchNorm', layer_collector=None):
    l = tf.layers.batch_normalization(input, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def Softmax(input, kwargs={}, name='_Softmax', layer_collector=None):
    l = tf.nn.softmax(input, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l