from nltk.corpus import wordnet as wn


n = wn.synset('procession.n.02')
print(n.definition())
for h in n.hypernym_paths():
    print(h)