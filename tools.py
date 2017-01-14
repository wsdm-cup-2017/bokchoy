import numpy as np
import pickle
import math
from sklearn.externals import joblib


def map_lin(scores):
    max_score = max(scores)
    if max_score == 0:
        return [0] * len(scores)
    return [int(score / max_score * 7) for score in scores]


def map_log(scores):
    max_score = max(scores)
    return [max(int(math.log2(score/max_score * (2 ** 7))), 0) for score in scores]


def map_log_round(scores):
    max_score = max(scores)
    return [max(round(math.log2(score/max_score * (2 ** 7))), 0) for score in scores]


def map_log_lin(scores):
    scores = [1/-math.log10(score) for score in scores]
    return map_lin(scores)


def map_scale(scores):
    return [int(score * 8 - 0.00001) for score in scores]


def safe_load(path, load_type='pkl'):
    if load_type == 'pkl':
        return pickle.load(open(path, 'rb'))
    elif load_type == 'npy':
        return np.load(path)
    elif load_type == 'model':
        return joblib.load(path)
    else:
        print('Wrong Load Type')


def quick_dump(obj, path, dump_type):
    if dump_type == 'pkl':
        pickle.dump(obj, open(path, 'wb'))
    elif dump_type == 'npy':
        np.save(path, obj)
    elif dump_type == 'model':
        joblib.dump(obj, path)
    else:
        print('Wrong Dump Type')
