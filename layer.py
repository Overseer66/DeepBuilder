import inspect
import tensorflow as tf
import numpy as np
from .util import *

def reshape(input, shape, name='_Reshape', layer_collector=None, param_collector=None):
    l = tf.reshape(input, shape, name=name)
    safe_append(layer_collector, l)

    return l

def flatten(input, name='_Dense', layer_collector=None, param_collector=None):
    try:
        flat_size = int(np.prod(input.get_shape()[1:]))
    except:
        flat_size = tf.reduce_prod(tf.shape(input)[1:])

    l = tf.reshape(input, (-1, flat_size), name=name)
    safe_append(layer_collector, l)

    return l

def fully_connected_layer(
        input,
        output_size,
        initializer=tf.truncated_normal_initializer(stddev=2e-2),
        activation=tf.nn.relu,
        batch_norm_param=None,
        name='_Dense',
        layer_collector=None,
        param_collector=None
):
    w = tf.get_variable(name + '_weight', [input.get_shape()[1], output_size], initializer=initializer)
    safe_append(param_collector, w)
    b = tf.get_variable(name + '_bias', [output_size], initializer=initializer, dtype=tf.float32)
    safe_append(param_collector, b)

    l = tf.nn.bias_add(tf.matmul(input, w), b, name=name + '_layer')
    safe_append(layer_collector, l)

    if batch_norm_param != None:
        l = tf.layers.batch_normalization(l, **batch_norm_param, name=name + '_batch_norm')
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name + '_' + activation.__name__)
        safe_append(layer_collector, l)

    return l


def conv_2d(
        input,
        kernel_size,
        stride_size=[1, 1, 1, 1],
        padding='SAME',
        initializer=tf.truncated_normal_initializer(stddev=2e-2),
        activation=tf.nn.relu,
        batch_norm_param=None,
        name='_Conv2D',
        layer_collector=None,
        param_collector=None
):
    if type(kernel_size) == tuple: kernel_size = list(kernel_size)
    if kernel_size[2] == -1: kernel_size = [kernel_size[0], kernel_size[1], input.get_shape()[-1], kernel_size[3]]

    w = tf.get_variable(name + '_weight', kernel_size, initializer=initializer)
    safe_append(param_collector, w)
    b = tf.get_variable(name + '_bias', kernel_size[-1], initializer=initializer)
    safe_append(param_collector, b)
    c = tf.nn.conv2d(input, w, strides=stride_size, padding=padding)

    l = tf.nn.bias_add(c, b, name=name + '_layer')
    safe_append(layer_collector, l)

    if batch_norm_param != None:
        l = tf.layers.batch_normalization(l, **batch_norm_param, name=name + '_batch_norm')
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name + '_' + activation.__name__)
        safe_append(layer_collector, l)

    return l


def deconv_2d(
        input,
        kernel_size,
        output_shape,
        stride_size=[1, 1, 1, 1],
        padding='SAME',
        initializer=tf.truncated_normal_initializer(stddev=2e-2),
        activation=tf.nn.relu,
        batch_norm_param=None,
        name='_Deconv2D',
        layer_collector=None,
        param_collector=None
):
    if type(kernel_size) == tuple: kernel_size = list(kernel_size)
    if kernel_size[2] == -1: kernel_size = [kernel_size[0], kernel_size[1], output_shape[-1], kernel_size[3]]
    if kernel_size[3] == -1: kernel_size = [kernel_size[0], kernel_size[1], kernel_size[2], input.get_shape()[-1]]

    if type(output_shape) == tuple: output_shape = list(output_shape)
    if output_shape[0] == -1: output_shape = [tf.shape(input)[0], output_shape[1], output_shape[2], output_shape[3]]

    w = tf.get_variable(name + '_weight', kernel_size, initializer=initializer)
    safe_append(param_collector, w)
    b = tf.get_variable(name + '_bias', kernel_size[-2], initializer=initializer)
    safe_append(param_collector, w)
    c = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=stride_size, padding=padding)

    l = tf.nn.bias_add(c, b, name=name + '_layer')
    safe_append(layer_collector, l)

    if batch_norm_param != None:
        l = tf.layers.batch_normalization(l, **batch_norm_param, name=name + '_batch_norm')
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name + '_' + activation.__name__)
        safe_append(layer_collector, l)


    return l


def max_pool(
        input,
        kernel_size=[1, 2, 2, 1],
        stride_size=[1, 2, 2, 1],
        padding='SAME',
        name='_MaxPooling',
        layer_collector=None,
        param_collector=None
):
    l = tf.nn.max_pool(input, kernel_size, stride_size, name)
    safe_append(layer_collector, l)

    return l


def repeat(
        input,
        layer,
        count,
        name='_Repeat',
        kwargs={},
        layer_collector=None,
        param_collector=None,
):
    if 'layer_collector' in inspect.signature(layer).parameters.keys():
        kwargs['layer_collector'] = layer_collector
    if 'param_collector' in inspect.signature(layer).parameters.keys():
        kwargs['param_collector'] = param_collector

    l = input
    for idx in range(count):
        with tf.variable_scope(name+'_'+str(idx+1)):
            l = layer(l, **kwargs)
            safe_append(layer_collector, l)

    return l


def residual(
        input,
        layer,
        step=2,
        activation=tf.nn.relu,
        name='_Residual',
        kwargs={},
        layer_collector=None,
        param_collector=None,
):
    if 'layer_collector' in inspect.signature(layer).parameters.keys():
        kwargs['layer_collector'] = layer_collector
    if 'param_collector' in inspect.signature(layer).parameters.keys():
        kwargs['param_collector'] = param_collector

    with tf.variable_scope(name):
        l = repeat(input, layer, step, **kwargs)
        l = tf.add(l, input)

    if activation:
        l = activation(l, name=name + '_' + activation.__name__)
        safe_append(layer_collector, l)

    return l


def dense_connection(
        input,
        layer,
        activation=None,
        name='_DenseConnect',
        kwargs={},
        layer_collector=None,
        param_collector=None,
):
    if 'layer_collector' in inspect.signature(layer).parameters.keys():
        kwargs['layer_collector'] = layer_collector
    if 'param_collector' in inspect.signature(layer).parameters.keys():
        kwargs['param_collector'] = param_collector

    with tf.variable_scope(name):
        l = layer(input, **kwargs)
        l = tf.concat((input, l), axis=3)
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name + '_' + activation.__name__)
        safe_append(layer_collector, l)

    return l


def dense_block(
        input,
        iterate,
        activation=tf.nn.relu,
        name='_DenseBlock',
        kwargs={},
        layer_collector=None,
        param_collector=None,
):
    kwargs['layer_collector'] = layer_collector
    kwargs['param_collector'] = param_collector

    with tf.variable_scope(name):
        l = repeat(input, dense_connection, iterate, **kwargs)

    if activation:
        l = activation(l, name=name + '_' + activation.__name__)
        safe_append(layer_collector, l)

    return l