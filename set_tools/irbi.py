from sklearn.model_selection import StratifiedKFold
from . import get_Xy, select_subcsv, grouped, flatten
from statistics import mean
from math import ceil

def group_data(l):
    return [[[i[0] for i in g2] for g2 in grouped(g, 2)] for g in grouped(l, 1)]

def get_stratified_kfold(csvcontent, k):
    struct = group_data(csvcontent)
    X, y = get_Xy(struct)
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    select_subset = lambda s : select_subcsv(csvcontent, 0, flatten(s))
    for train, test in skf.split(y, y):
        testing_set = [X[i] for i in test]
        training_set = [X[i] for i in train]
        yield (select_subset(training_set), select_subset(testing_set))

def get_subsets(csvcontent, min_img, step):
    struct = group_data(csvcontent)
    for i in range(0, 101, step):
        nb = lambda l : max(ceil(l*i/100), min_img)
        subset = flatten([flatten(c[:nb(len(c))]) for c in struct])
        yield select_subcsv(csvcontent, 0, subset)
