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

person2id__dict = os.path.join(data_root, folder, 'person2id.dict')
id2person__dict = os.path.join(data_root, folder, 'id2person.dict')
id2prof__dict = os.path.join(data_root, folder, 'id2prof.dict')
person_profs__dict = os.path.join(data_root, folder, 'person_profs_in_wsdm.dict')
prof_persons__dict = os.path.join(data_root, folder, 'prof_persons_wsdm.dict')
pos_prof_persons_filter__dict = os.path.join(data_root, folder, 'pos_profs_person_filter.dict')
neg_prof_persons_filter__dict = os.path.join(data_root, folder, 'neg_profs_person_filter.dict')
pos_nation_persons_triples_with_one__dict = os.path.join(data_root, folder, 'pos_nation_persons_triple.dict')
neg_nation_persons_triple_all__dict = os.path.join(data_root, folder, 'neg_nation_person_triple_all.dict')
nation_persons__dict = os.path.join(data_root, folder, 'nation_persons.dict')
person_nations__dict = os.path.join(data_root, folder, 'person_nations.dict')
pname_popularity__dict = os.path.join(data_root, folder, 'pname_popularity.dict')
nationalities__set = os.path.join(data_root, folder, 'nationalities.set')
professions__set = os.path.join(data_root, folder, 'professions.set')
prof_persons_with_one__dict = os.path.join(data_root, folder, 'persons_in_prof_one_wsdm.dict')
prof_name2id__dict = os.path.join(data_root, folder, 'prof_name2id_wsdm.dict')
nation_name2id__dict = os.path.join(data_root, folder, 'nation_name2id_wsdm.dict')
pos_text_prof__list = os.path.join(data_root, folder, 'pos_text_prof.list')
description__dict = os.path.join(data_root, folder, 'description.dict')
nation_id2name__dict = os.path.join(data_root, folder, 'nation_id2nam.dict')
country_name = os.path.join(data_root, folder, 'country.txt')
pos_nations_text = os.path.join(data_root, folder, 'pos_nations.dict')
neg_nations_text = os.path.join(data_root, folder, 'neg_nations.dict')
person_text = os.path.join(data_root, folder, 'person_text.dict')



def attr_name2id(task):
    if task == 'prof':
        return prof_name2id__dict
    else:
        return nation_name2id__dict
