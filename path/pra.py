from path.conf import data_root
import os

folder = 'pra'
edgeList = os.path.join(data_root, folder, 'kg', 'edgeList')
nodeList = os.path.join(data_root, folder, 'kg', 'nodeType')
edgeType = os.path.join(data_root, folder, 'kg', 'edgeType')
pra_prof__set = os.path.join(data_root, folder, 'pra_prof.set')
prof_triples__list = os.path.join(data_root, folder, 'prof_triples.list')
person_triples_count__dict = os.path.join(data_root, folder, 'person_triples_count.dict')
pos_triples_count__dict = os.path.join(data_root, folder, 'pos_triple_count.dict')
pos_prof_triples__dict = os.path.join(data_root, folder, 'pos_prof_triples')
neg_prof_triples__dict = os.path.join(data_root, folder, 'neg_prof_triples')
pos_nation_triples = os.path.join(data_root, folder, 'nation_pos_triples')
neg_nation_triples = os.path.join(data_root, folder, 'nation_neg_triples')
path__dict = os.path.join(data_root, folder, 'path_dict_100_prof_rel')


def triple_file(task, n):
    return os.path.join(data_root, folder, 'kg', 'in', task, str(n))


def path_file(task, n):
    return os.path.join(data_root, folder, 'kg', 'out', task, str(n))


def out_path_file(sub_folder, n):
    return os.path.join(data_root, folder, 'kg', 'out', sub_folder, str(n))


def out_path_files(sub_folder):
    par = os.path.join(data_root, folder, 'kg', 'out', sub_folder)
    files = os.listdir(par)
    return [out_path_file(sub_folder, file) for file in files]



def rf_path(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'rf')


def path_feature(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'path_feature')


def sel_idx(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'sel_idx')


def X_train(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'X_train.npy')


def y_train(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'y_train.npy')


def X_val(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'X_val.npy')


def y_val(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'y_val.npy')


def X_test(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'X_test.npy')

def test_order(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'test_order')


def path_dicts_train(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'path_dicts_train.dict')


def path_dicts_test(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'path_dicts_test.dict')


def result(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'result')


def X_score(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'X_score.npy')


def score_order(task):
    this_folder = os.path.join(data_root, folder, task)
    if not os.path.exists(this_folder):
        os.makedirs(this_folder)
    return os.path.join(this_folder, 'score_order')





