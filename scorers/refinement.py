from nltk.corpus import wordnet as wn
import tools as tools
import path.wsdm as wsdm_path
import nltk


def get_plural(word):
    if word[-1:] == 'y':
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word[-2:] == 'an':
        return word[:-2] + 'en'
    else:
        return word + 's'


def get_hyponyms(prof):
    hyponym_set = set()
    synsets = wn.synsets(prof, pos=wn.NOUN)
    for synset in synsets:
        hyponyms = synset.hyponyms()
        for hyponym in hyponyms:
            names = hyponym.lemma_names()
            for name in names:
                hyponym_set.add(name)
    return hyponym_set


def get_syn(prof):
    syn_set = set()
    synsets = wn.synsets(prof)
    for synset in synsets:
        names = synset.lemma_names()
        for name in names:
            syn_set.add(name)
    return syn_set


def get_all_prof_qdict():
    prof_set = tools.safe_load(wsdm_path.professions__set, load_type='pkl')
    prof_qdict = {}
    for prof in prof_set:
        prof_qdict[prof] = {}
        tmp_prof = '_'.join(prof.split(' '))
        tmp_syn_set = get_syn(tmp_prof)
        tmp_hyponym_set = get_hyponyms(tmp_prof)
        syn_set = set()
        hyponym_set = set()
        for syn in tmp_syn_set:
            name = ' '.join(syn.split('_')).lower()
            syn_set.add(name)
            syn_set.add(get_plural(name))
        name = prof.lower()
        syn_set.add(name)
        syn_set.add(get_plural(name))
        for hyponym in tmp_hyponym_set:
            name = ' '.join(hyponym.split('_')).lower()
            hyponym_set.add(name)
            hyponym_set.add(get_plural(name))
        prof_qdict[prof]['syn'] = syn_set
        prof_qdict[prof]['hyponyms'] = hyponym_set
    return prof_qdict


def find_nation_kw(person, person_nations, person_text, nation_qdict):
    result = {}
    text = person_text[person]
    sent_token = nltk.tokenize.sent_tokenize(text)
    first = len(sent_token[0])
    for nation, qlist in nation_qdict.items():
        if nation not in person_nations[person]:
            continue
        for _nation in qlist:
            start = text.find(_nation)
            if start > 0 and not text[start - 1].isalpha() and not text[start + len(_nation)].isalpha():
                if nation not in result.keys():
                    result[nation] = []
                result[nation].append(start)
    for nation, nation_list in result.items():
        result[nation] = min(nation_list)
    return result, first


def find_kw(person, person_prfos, person_text, prof_qdit):
    result = {'syn': {}, 'hyponym': {}}
    text = person_text[person].replace('\n', '')
    sent_token = nltk.tokenize.sent_tokenize(text)
    first = len(sent_token[0])
    result['first'] = first
    text = text.lower()
    for prof, qdict in prof_qdit.items():
        if prof not in person_prfos[person]:
            continue
        syns = qdict['syn']
        hyponyms = qdict['hyponyms']
        for syn in syns:
            start = text.find(syn)
            if start > 0 and not text[start-1].isalpha() and not text[start + len(syn)].isalpha():
                if prof not in result['syn'].keys():
                    result['syn'][prof] = []
                result['syn'][prof].append(start)

        for hyponym in hyponyms:
            start = text.find(hyponym)
            if start > 0 and not text[start-1].isalpha() and not text[start + len(hyponym)].isalpha():
                if prof not in result['hyponym'].keys():
                    result['hyponym'][prof] = []
                result['hyponym'][prof].append(start)
    syn = result['syn']
    hyponyms = result['hyponym']
    prof_pos = {}
    for prof, pos_list in syn.items():
        if prof not in prof_pos.keys():
            prof_pos[prof] = []
        prof_pos[prof].extend(pos_list)
    for prof, pos_list in hyponyms.items():
        if prof not in prof_pos.keys():
            prof_pos[prof] = []
        prof_pos[prof].extend(pos_list)
    for prof, pos_list in prof_pos.items():
        prof_pos[prof] = min(pos_list)
    return prof_pos, first


def predict_prof(result, triples, person_profs, person_text, prof_qdict):
    rule_result = {}
    problem = []
    for person, prof in triples:
        if person not in rule_result.keys():
            rule_result[person] = {}
        rule_result[person][prof] = result[person][prof]

    for person, profs in rule_result.items():
        try:
            kw_dict, first = find_kw(person, person_profs, person_text, prof_qdict)
            for prof, pos in kw_dict.items():
                if rule_result[person][prof] < 5 and pos <= first:
                    rule_result[person][prof] = 5
        except Exception:
            problem.append(person)
            continue
    return rule_result, problem


def predict_nation(result, triples, person_nations, person_text, nation_qdict):
    rule_result = {}
    problem = []
    update = 0
    for person, nation in triples:
        if person not in rule_result.keys():
            rule_result[person] = {}
        rule_result[person][nation] = result[person][nation]

    for person in rule_result.keys():
        try:
            kw_dict, first = find_nation_kw(person, person_nations, person_text, nation_qdict)
            for nation, pos in kw_dict.items():
                if rule_result[person][nation] < 5 and pos <= first:
                    rule_result[person][nation] = 5
                    update += 1
                for nation in rule_result[person].keys():
                    if nation not in kw_dict.keys():
                        rule_result[person][nation] = 2
        except Exception:
            problem.append(person)
            continue
    return rule_result, problem, update