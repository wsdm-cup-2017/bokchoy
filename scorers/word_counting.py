from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tools import quick_dump, safe_load
from reader import get_type_text, get_person_text
from tqdm import tqdm
import path.words_counting as path
import numpy as np


def calculate_weight(type_tfidf):
    idx = np.argsort(type_tfidf.toarray()[0])[::-1]
    score_array = np.zeros(len(idx))
    for i, idx in enumerate(idx):
        score_array[idx] = 1 / (i + 1)
    return type_tfidf


def fit(task, type_list, corpus_accessor):
    type_tfidf = TfidfVectorizer(stop_words='english', max_features=100000, min_df=2)
    words_weights = type_tfidf.fit_transform(get_type_text(type_list))
    for person_type, tfidf in zip(type_list, words_weights):
        quick_dump(calculate_weight(tfidf).toarray(), path.type_weight(person_type, task), dump_type='npy')
        # print(person_type)
    quick_dump(type_tfidf, path.type_tfidf(task), dump_type='model')


def scoring(task, person_cnt_vec, person, person_type):
    X = person_cnt_vec.fit_transform(get_person_text(person))
    return np.dot(X.toarray()[0], safe_load(path.type_weight(person_type, task), load_type='npy')[0])


def predict(task, triples):
    result = {}
    problem = []
    type_tfidf = safe_load(path.type_tfidf(task), load_type='model')
    person_cnt_vec = CountVectorizer(vocabulary=type_tfidf.vocabulary_)
    for person, person_type in tqdm(triples):
        try:
            if not person in result.keys():
                result[person] = {}
            result[person][person_type] = scoring(task, person_cnt_vec, person, person_type)
        except Exception as e:
            problem.append([person, person_type, e])
    return result, problem


def dump_result_problem(self, task, result, problem):
    quick_dump(result, path.result(task), dump_type='pkl')
    quick_dump(problem, path.problem(task), dump_type='pkl')
