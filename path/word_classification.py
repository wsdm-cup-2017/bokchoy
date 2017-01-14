from path.conf import data_root
import os

folder = 'word_classification'


def train_feature(task, person_type):
    return os.path.join(data_root, folder, task, 'feature', person_type)


def type_tfidf(task, person_type):
    return os.path.join(data_root, folder, task, 'feature', person_type, 'vectorizer.pkl')


def model(task, attr):
    return os.path.join(data_root, folder, task, 'model', attr)


def result(task):
    return os.path.join(data_root, folder, task, 'result')


def problem(task):
    return os.path.join(data_root, folder, task, 'problem')
