def weighted_int_averaging(triples, scores, weights):
    ret = {}
    for person, attr in triples:
        avg = 0
        norm = 0
        do = 0
        write_score = 0
        for result, weight in zip(scores, weights):
            if person in result.keys() and attr in result[person].keys():
                avg += weight * result[person][attr]
                norm += weight
                do += 1
        if do > 0:
            avg /= norm
            write_score = int(avg)
        if person not in ret.keys():
            ret[person] = {}
        ret[person][attr] = write_score
    return ret
