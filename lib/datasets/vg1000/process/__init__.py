from nltk.corpus import wordnet as wn

syns = wn.synsets('table')
for syn in syns:
    print('====')
    print(syn.name()+':'+syn.definition())
    for h in syn.hypernym_paths():
        print(h)