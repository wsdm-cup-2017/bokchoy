import numpy as np
from tools import quick_dump
from reader import get_persons_text, get_types_text
from sklearn.feature_extraction.text import TfidfVectorizer
import path.word_mle as path


def vectorize(persons):
    person_tf_idf = TfidfVectorizer(max_features=20000, stop_words='english')
    person_words_probs = person_tf_idf.fit_transform(get_persons_text(persons))
    prof_tf_idf = TfidfVectorizer(vocabulary=person_tf_idf.vocabulary_)
    attr_words_probs = prof_tf_idf.fit_transform(get_types_text(person_type_list))
    return person_words_probs, attr_words_probs


def optimize(self, person_attrs_prob, person_words_prob, attr_words_prob, delta):
    previous = np.zeros(person_attrs_prob.shape)
    while np.linalg.norm(previous - person_attrs_prob) > delta:
        # E step
        attr_words_prob_trans = attr_words_prob.transpose()
        shape = attr_words_prob_trans.shape
        numerator = attr_words_prob_trans * np.broadcast_to(person_attrs_prob, shape)
        denominator = np.broadcast_to(np.dot(attr_words_prob_trans, person_attrs_prob).reshape(shape[0], 1), shape)
        smooth = np.min(denominator[~(denominator == 0)])
        words_prof_prob = numerator / (denominator + smooth)

        # M step
        previous = person_attrs_prob
        person_attrs_prob = np.dot(words_prof_prob.transpose(), person_words_prob)
        person_attrs_prob = person_attrs_prob / np.sum(person_attrs_prob)
    return person_attrs_prob


def fit(person_types, person_words_probs, type_words_probs, person_order_list, person_type_list, delta=0.0001):
    person_attrs_probs = {}
    problem = []
    for person, types in person_types.items():
        try:
            idx = person_order_list.index(person)
            person_words_prob = person_words_probs[idx]
            person_attrs_prob = np.array([1 / len(types)] * len(types))
            person_type_idx = []
            for person_type in types:
                person_type_idx.append(person_type_list.index(person_type))
            attr_words_prob = type_words_probs[np.array(person_type_idx)]
            person_attrs_prob = optimize(person_attrs_prob,
                                         person_words_prob.toarray()[0],
                                         attr_words_prob.toarray(),
                                         delta)
            person_attrs_probs[person] = {}
            for person_type, score in zip(types, person_attrs_prob):
                if not person_type == 'Pseudo Profession':
                    person_attrs_probs[person][person_type] = score
        except Exception as e:
            problem.append([person, e])
    return person_attrs_probs, problem
