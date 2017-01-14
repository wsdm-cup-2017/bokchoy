from path.conf import data_root
import os

folder = 'wsdm'
persons = os.path.join(data_root, folder, 'map-wiki2fb')
nationality__kb = os.path.join(data_root, folder, 'nationality.all')
nationalities = os.path.join(data_root, folder, 'nationality.list')
profession__kb = os.path.join(data_root, folder, 'profession.all')
professions = os.path.join(data_root, folder, 'profession.list')
nationality__train = os.path.join(data_root, folder, 'nationality.train')
profession__train = os.path.join(data_root, folder, 'profession.train')
wiki_sentences = os.path.join(data_root, folder, 'wiki-sentences')
