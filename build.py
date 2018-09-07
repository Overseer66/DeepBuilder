import inspect
import tensorflow as tf


class Builder(object):
    def __init__(self, architecture):
        self.architecture = architecture

    def __call__(self, input, scope, reuse=False):
        layer_collector = []
        param_collector = []

        last_layer = input

        with tf.variable_scope(scope, reuse=reuse):
            for idx, (layer_dict) in enumerate(self.architecture):
                method = layer_dict['method']
                args = layer_dict['args']
                kwargs = layer_dict['kwargs']
                with tf.variable_scope('Layer' + str(idx), reuse=reuse):
                    if 'layer_collector' in inspect.signature(method).parameters.keys():
                        kwargs['layer_collector'] = layer_collector
                    if 'param_collector' in inspect.signature(method).parameters.keys():
                        kwargs['param_collector'] = param_collector
                    last_layer = method(input=last_layer, *args, **kwargs)

        return last_layer, layer_collector, param_collector