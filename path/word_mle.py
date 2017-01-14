from path.conf import data_root
import os

folder = 'word_mle'
person_words_probs = os.path.join(data_root, folder, 'person_words_probs')
attr_words_probs = os.path.join(data_root, folder, 'prof_words_probs')
person_attr_probs = os.path.join(data_root, folder, 'person_attr_probs')


def result(task):
    return os.path.join(data_root, folder, task, 'result')


def problem(task):
    return os.path.join(data_root, folder, task, 'problem')

