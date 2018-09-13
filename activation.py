import tensorflow as tf
from .util import *

def Dropout(input, keep_prob, name='_Dropout', layer_collector=None, *args, **kwargs):
    l = tf.nn.dropout(input, keep_prob, *args, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def Activation(input, activation, name=None, layer_collector=None, *args, **kwargs):
    name = activation.name if name == None else name
    l = activation(input, *args, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def LeakyReLU(input, leak=0.2, name='_LeakyReLU', layer_collector=None):
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        l = f1 * input + f2 * abs(input)
        safe_append(layer_collector, l)
        return l


def Scale(input, width, height, name='_Scale', layer_collector=None, *args, **kwargs):
    l = tf.image.resize_nearest_neighbor(input, (width, height), *args, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def BatchNorm(input, name='_BatchNorm', layer_collector=None, *args, **kwargs):
    l = tf.layers.batch_normalization(input, *args, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def Softmax(input, name='_Softmax', layer_collector=None, *args, **kwargs):
    l = tf.nn.softmax(input, *args, **kwargs, name=name)
    safe_append(layer_collector, l)
    return l


def Transpose(input, permutation, name='_Transpose', layer_collector=None):
    l = tf.transpose(input, permutation)
    safe_append(layer_collector, l)
    return l



