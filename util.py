import tensorflow as tf
import collections

def AppendInputs(input, layer_collector):
    for source in input:
        safe_append(layer_collector, source)
    return input


def LayerIndexer(input, indices):
    l = []
    for idx in indices:
        l.append(input[idx])
    if len(l) == 1:
        l = l[0]
    return l


def LayerSelector(input, names, layer_collector):
    l = []
    for name in names:
        layer = SearchLayer(layer_collector, name)
        if layer != None:
            l.append(layer)
    if len(l) == 1:
        l = l[0]
    return l


def SearchLayer(layers, name):
    for layer in layers:
        if type(layer) != list and type(layer) != tuple:
            fullname = layer.name.split(':')[0]
            idx = fullname.find(name)
            if idx != -1:
                if idx == 0 or fullname[idx-1] != '/':
                    return layer
        else:
            layer = SearchLayer(layer, name)
            if layer != None:
                return layer


def safe_append(l, v):
    if type(l) == list:
        if v not in l:
            l.append(v)


class ScopeSelector(object):
    def __init__(self, scope, reuse, *args, **kwargs):
        self.scope = tf.variable_scope(scope, reuse=reuse) if scope else None
    def __enter__(self):
        if self.scope:
            return self.scope.__enter__()
        else:
            return None
    def __exit__(self, exc_type, exc_value, traceback):
        if self.scope:
            return self.scope.__exit__(exc_type, exc_value, traceback)
        else:
            return False


