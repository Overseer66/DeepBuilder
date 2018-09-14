import tensorflow as tf

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