import inspect
import tensorflow as tf
import numpy as np
from .util import *


def reshape(input, shape, name='Reshape', layer_collector=None):
    l = tf.reshape(input, shape, name=name)
    safe_append(layer_collector, l)

    return l


def flatten(input, name='Dense', layer_collector=None):
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
        name='Dense',
        weight_name = 'weights',
        bias_name = 'biases',
        layer_collector=None,
        param_collector=None
    ):
    with ScopeSelector(name, False):
        w = tf.get_variable(weight_name, [input.get_shape()[-1], output_size], initializer=initializer)
        b = tf.get_variable(bias_name, [output_size], initializer=initializer, dtype=tf.float32)
    safe_append(param_collector, w)
    safe_append(param_collector, b)

    l = tf.nn.bias_add(tf.matmul(input, w), b, name=name if not activation else None)
    safe_append(layer_collector, l)

    if batch_norm_param != None:
        l = tf.layers.batch_normalization(l, **batch_norm_param, name=name + '_batch_norm')
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name)
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
        name='Conv2D',
        weight_name = 'weights',
        bias_name = 'biases',
        layer_collector=None,
        param_collector=None
    ):
    if type(kernel_size) == tuple: kernel_size = list(kernel_size)
    if kernel_size[2] == -1: kernel_size = [kernel_size[0], kernel_size[1], input.get_shape()[-1], kernel_size[3]]

    with ScopeSelector(name, False):
        w = tf.get_variable(weight_name, kernel_size, initializer=initializer)
        b = tf.get_variable(bias_name, kernel_size[-1], initializer=initializer)
    safe_append(param_collector, w)
    safe_append(param_collector, b)
    c = tf.nn.conv2d(input, w, strides=stride_size, padding=padding)

    l = tf.nn.bias_add(c, b, name=name if not activation else None)
    safe_append(layer_collector, l)

    if batch_norm_param != None:
        l = tf.layers.batch_normalization(l, **batch_norm_param, name='batch_norm')
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name)
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
        name='Deconv2D',
        weight_name = 'weights',
        bias_name = 'biases',
        layer_collector=None,
        param_collector=None
    ):
    if type(kernel_size) == tuple: kernel_size = list(kernel_size)
    if kernel_size[2] == -1: kernel_size = [kernel_size[0], kernel_size[1], output_shape[-1], kernel_size[3]]
    if kernel_size[3] == -1: kernel_size = [kernel_size[0], kernel_size[1], kernel_size[2], input.get_shape()[-1]]

    if type(output_shape) == tuple: output_shape = list(output_shape)
    if output_shape[0] == -1: output_shape = [tf.shape(input)[0], output_shape[1], output_shape[2], output_shape[3]]

    with ScopeSelector(name, False):
        w = tf.get_variable(weight_name, kernel_size, initializer=initializer)
        b = tf.get_variable(bias_name, kernel_size[-2], initializer=initializer)
    safe_append(param_collector, w)
    safe_append(param_collector, b)
    c = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=stride_size, padding=padding)

    l = tf.nn.bias_add(c, b, name=name if not activation else None)
    safe_append(layer_collector, l)

    if batch_norm_param != None:
        l = tf.layers.batch_normalization(l, **batch_norm_param, name='batch_norm')
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name)
        safe_append(layer_collector, l)


    return l


def max_pool(
        input,
        kernel_size=[1, 2, 2, 1],
        stride_size=[1, 2, 2, 1],
        padding='SAME',
        name='MaxPooling',
        layer_collector=None
    ):
    l = tf.nn.max_pool(input, kernel_size, stride_size, padding, name=name)
    safe_append(layer_collector, l)

    return l


def repeat(
        input,
        layer_dict,
        count,
        name='Repeat',
        layer_collector=None,
        param_collector=None,
    ):
    method = layer_dict['method']
    args = layer_dict['args']
    kwargs = layer_dict['kwargs']

    if 'layer_collector' in inspect.signature(method).parameters.keys():
        kwargs['layer_collector'] = layer_collector
    if 'param_collector' in inspect.signature(method).parameters.keys():
        kwargs['param_collector'] = param_collector

    l = input
    for idx in range(count):
        with tf.variable_scope(name+'_'+str(idx+1)):
            l = method(l, *args, **kwargs)
            safe_append(layer_collector, l)

    return l


def residual(
        input,
        layer_dict,
        step=2,
        activation=tf.nn.relu,
        name='Residual',
        layer_collector=None,
        param_collector=None,
    ):
    with ScopeSelector(name if not activation else None, False):
        l = repeat(input, layer_dict, step, layer_collector=layer_collector, param_collector=param_collector)
        l = tf.add(l, input)

    if activation:
        l = activation(l, name=name)
        safe_append(layer_collector, l)

    return l


def dense_connection(
        input,
        layer_dict,
        activation=None,
        name='DenseConnect',
        layer_collector=None,
        param_collector=None,
    ):
    method = layer_dict['method']
    args = layer_dict['args']
    kwargs = layer_dict['kwargs']

    if 'layer_collector' in inspect.signature(method).parameters.keys():
        kwargs['layer_collector'] = layer_collector
    if 'param_collector' in inspect.signature(method).parameters.keys():
        kwargs['param_collector'] = param_collector

    with ScopeSelector(name if not activation else None, False):
        l = method(input, *args, **kwargs)
        l = tf.concat((input, l), axis=3)
        safe_append(layer_collector, l)

    if activation:
        l = activation(l, name=name)
        safe_append(layer_collector, l)

    return l


def dense_block(
        input,
        layer_dict,
        iterate,
        activation=tf.nn.relu,
        name='DenseBlock',
        layer_collector=None,
        param_collector=None,
    ):
    layer_dict = {
        'method': dense_connection,
        'args': (),
        'kwargs': {
            'layer_dict': layer_dict
        }
    }

    with ScopeSelector(name if not activation else None, False):
        l = repeat(input, layer_dict, iterate, layer_collector=layer_collector, param_collector=param_collector)

    if activation:
        l = activation(l, name=name)
        safe_append(layer_collector, l)

    return l



