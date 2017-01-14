from tools import quick_dump, safe_load
from reader import get_person_text, get_persons_text
import path.word_classification as path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm
import numpy as np
import os


def vectorize(task, type_pos_neg_persons):
    for person_type, persons in tqdm(type_pos_neg_persons.items()):
        type_tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
        train_feature = type_tfidf.fit_transform(get_persons_text(persons))
        quick_dump(path.type_tfidf(task, person_type), dump_type='model')
        quick_dump(path.train_feature(task, person_type), dump_type='pkl')


def train(task, type_pos_neg_persons, type_person_label):
    for person_type, persons in tqdm(type_pos_neg_persons.items()):
        X = safe_load(path.train_feature(task, person_type), load_type='pkl')
        y = np.array(type_person_label[person_type])
        lr = LogisticRegressionCV(cv=5, solver='liblinear')
        lr.fit(X, y)
        if not os.path.exists(path.model(task, person_type)):
            os.makedirs(path.model(task, person_type))
        quick_dump(lr, path.model(task, person_type), dump_type='model')


def predict(task, triples):
    result = {}
    problem = []
    for person, person_type in tqdm(triples):
        try:
            vectorizer = safe_load(path.vectorizer(task, person_type), load_type='model')
            lr = safe_load(path.model(task, person_type), load_type='model')
            X_test = vectorizer.transform(get_person_text(person))
            score = lr.predict_proba(X_test)
            if not person in result.keys():
                result[person] = {}
            result[person][person_type] = score[0][1]
        except Exception as e:
            problem.append([person, person_type, e])
    return result, problem
