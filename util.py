

def safe_append(l, v):
    if type(l) == list:
        if v not in l:
            l.append(v)