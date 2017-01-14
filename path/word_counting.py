from path.conf import data_root
import os

folder = 'word_counting'


def type_weight(task, person_type):
    return os.path.join(data_root, folder, task, 'weight', person_type+'_wt.npy')


def type_tfidf(task):
    return os.path.join(data_root, folder, task, 'model', 'attr_tf_idf.model')


def result(task):
    return os.path.join(data_root, folder, task, 'result')


def problem(task):
    return os.path.join(data_root, folder, task, 'problem')