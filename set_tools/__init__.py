import csv
from itertools import groupby

def compute_class_weights(y, training_names):
    y = get_class_ids(y, training_names)
    nb_classes = len(training_names)
    cards = [y.count(c) for c in range(nb_classes)]
    ma = max(cards)
    return {i:ma/c for i, c in enumerate(cards)}

def get_class_ids(y, training_names):
    return [training_names.index(el) for el in y]

def select_subcsv(csvcontent, key, subset):
    return [i for i in csvcontent if i[key] in subset]

def remove(l, i):
    return l[:i] + l[(i + 1):]

def grouped(l, key):
    fn = lambda n : n[key]
    g = groupby(sorted(l, key=fn), key=fn)
    return [[remove(img, key) for img in i[1]] for i in g]
    
def flatten(struct):
    return [j for i in struct for j in i]

def get_Xy(l):
    return (flatten(out) for out in zip(*[(clas, [i]*len(clas)) for i, clas in enumerate(l)]))

def get_class_list(csvcontent):
    return sorted(list(set([l[1] for l in csvcontent])))

def write_csv(filepath, newcsv):
    with open(filepath, "w+") as f:
        wr = csv.writer(f)
        for r in newcsv:
            wr.writerow(r)

def get_set_tuple(csvcontent):
    return tuple(zip(*[(l[0], l[1]) for l in csvcontent]))
