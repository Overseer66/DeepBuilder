import tensorflow as tf
from .util import *

def Dropout(input, keep_prob, name='Dropout', layer_collector=None, *args, **kwargs):
    l = tf.nn.dropout(input, keep_prob, *args, **kwargs, name=name)
    safe_append(layer_collector, l, name)
    return l


def Activation(input, activation, name=None, layer_collector=None, *args, **kwargs):
    name = activation.name if name == None else name
    l = activation(input, *args, **kwargs, name=name)
    safe_append(layer_collector, l, name)
    return l


def LeakyReLU(input, leak=0.2, name='LeakyReLU', layer_collector=None):
    with ScopeSelector(name, False):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        l = f1 * input + f2 * abs(input)
        safe_append(layer_collector, l, name)
        return l


def Scale(input, width, height, name='Scale', layer_collector=None, *args, **kwargs):
    l = tf.image.resize_nearest_neighbor(input, (width, height), *args, **kwargs, name=name)
    safe_append(layer_collector, l, name)
    return l


def BatchNorm(input, name='BatchNorm', layer_collector=None, *args, **kwargs):
    l = tf.layers.batch_normalization(input, *args, **kwargs, name=name)
    safe_append(layer_collector, l, name)
    return l


def Softmax(input, name='Softmax', layer_collector=None, *args, **kwargs):
    l = tf.nn.softmax(input, *args, **kwargs, name=name)
    safe_append(layer_collector, l, name)
    return l


def Transpose(input, permutation, name='Transpose', layer_collector=None):
    l = tf.transpose(input, permutation, name=name)
    safe_append(layer_collector, l, name)
    return l


def Concatenate(input, name='Concat', layer_collector=None):
    l = tf.concat(input, axis=len(input[0].shape)-1)
    safe_append(layer_collector, l, name)
    return l


def Add(input, name='Input', layer_collector=None):
    l = tf.add(*input)
    safe_append(layer_collector, l, name)
    return l