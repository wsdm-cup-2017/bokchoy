import random
import tools
import path.pra as pra
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def get_entity_set(triples):
    entity_set = set()
    for h, r, t in triples:
        entity_set.add(h)
        entity_set.add(t)
    return entity_set


def get_rel_set(triples):
    rel_set = set()
    for h, r, t in triples:
        rel_set.add(r)
    return rel_set


def get_rel_dict(rel_set):
    return {i: rel for i, rel in enumerate(rel_set)}


def get_rel_map(rel_dict):
    return {rel: i for i, rel in rel_dict.items()}


def get_ent_dict(entity_set):
    return {i: ent for i, ent in enumerate(entity_set)}


def get_ent_map(entity_dict):
    return {ent: i for i, ent in entity_dict.items()}


def gen_edge_type(num_rel, edge_type_file):
    with open(edge_type_file, 'w') as f:
        for i in range(num_rel):
            f.write("%d\t%d\n" % (i, i))


def gen_node_type(num_ent, node_type_file):
    with open(node_type_file, 'w') as f:
        for i in range(num_ent):
            f.write("%d\t%d\n" % (i, i))


def gen_edge_list(triples, ent_map, rel_map, edge_list_file):
    with open(edge_list_file, 'w') as f:
        for h, r, t in triples:
            f.write("%d\t%d\t%d\n" % (ent_map[h], ent_map[t], rel_map[r]))


def map_triples(triples, ent_map):
    return [[ent_map[h], ent_map[t]] for h, r, t in triples]


def gen_path_triples_file(triples, file_path, size=100):
    for i in range(0, len(triples), size):
        with open(file_path(int(i / size)), 'w') as f:
            for h, t in triples[i:i+size]:
                f.write("%d,%d\n" % (h, t))


def gen_pos(triples, rel):
    return list(filter(lambda e: e[1] == rel, triples))


def gen_neg(pos, person_attrs):
    attrs = set()
    for person, _attrs in person_attrs.items():
        for attr in _attrs:
            attrs.add(attr)
    neg = []
    for h, _, t in pos:
        re_attrs = list(attrs - person_attrs[h])
        neg.append([h, _, re_attrs[random.randrange(0, len(re_attrs))]])
    return neg


def path2dict(file):
    path_dict = {}
    with open(file) as f:
        s = f.read().split('+')[1:]
        for triple_path in s:
            triple_path = triple_path.split('\n')[:-1]
            triple = triple_path[1]
            path_dict[triple] = {}
            for path in triple_path[2:]:
                if path not in path_dict[triple].keys():
                    path_dict[triple][path] = 0
                path_dict[triple][path] += 1
    return path_dict


def get_path_dicts(sub_folder):
    out_files = pra.out_path_files(sub_folder)
    path_dicts = [path2dict(file) for file in out_files]
    path_dict = {}
    for file in path_dicts:
        for triple, paths in file.items():
            path_dict[triple] = paths
    return path_dict


def path2feature(path_dicts, triples, path_feature=None, min_pf=1):
    ent_map = tools.safe_load(path.entity_map, load_type='pkl')
    nation2id = tools.safe_load(path.nation_name2id__dict, load_type='pkl')
    person2id = tools.safe_load(path.person2id__dict, load_type='pkl')
    if not path_feature:
        path_feature = {}
        for path_dict in path_dicts.values():
            for path, cnt in path_dict.items():
                if path not in path_feature.keys():
                    path_feature[path] = 0
                path_feature[path] += 1
        path_feature = list(dict(list(filter(lambda e: e[1] > min_pf, path_feature.items()))).keys())

    X = []
    order = []
    lack = []
    for h, t in triples:
        try:
            h = ent_map[person2id[h]]
            t = ent_map[nation2id[t]]
            encode = ','.join([str(h), str(t)])
            paths = path_dicts[encode]
            tmp = []
            for path in path_feature:
                if path in paths.keys():
                    tmp.append(paths[path])
                else:
                    tmp.append(0)
            X.append(tmp)
            order.append([h, t])
        except Exception as e:
            lack.append([h, t])
    return np.array(X), order, path_feature, lack


def train_rf(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, oob_score=True)
    min_samples_splits = [2, 5, 10]
    max_features = ['auto', 'log2']
    best_min_samples_split = 2
    best_max_feature = 'auto'
    best_score = 0
    for min_samples_split in min_samples_splits:
        rf.set_params(min_samples_split=min_samples_split)
        rf.fit(X_train, y_train)
        y_score = rf.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_score)
        print("max samples splits: %d" % min_samples_split)
        print("validation ROC-AUC: %.4f" % auc_score)
        if auc_score >= best_score:
            best_min_samples_split = min_samples_split
            best_score = auc_score

    for max_feature in max_features:
        rf.set_params(max_features=max_feature)
        rf.fit(X_train, y_train)
        y_score = rf.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_score)
        print("max features: %s " % max_feature)
        print("validation ROC-AUC: %.4f" % auc_score)
        if auc_score >= best_score:
            best_max_feature = max_feature
            best_score = auc_score

    print("Best min samples split: %d" % best_min_samples_split)
    print("Bset max features : %s" % best_max_feature)
    rf.set_params(min_samples_split=best_min_samples_split, max_features=best_max_feature)
    rf.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    return rf


def get_pos_prof_triples(pos_prof_dict):
    triples = []
    for prof, persons in pos_prof_dict.items():
        for person in persons:
            triples.append([person, prof])
    return triples

