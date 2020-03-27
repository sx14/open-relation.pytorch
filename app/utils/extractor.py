from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree
from nltk.stem import WordNetLemmatizer

dep_parser = CoreNLPDependencyParser(url='http://0.0.0.0:9000')
pos_tagger = CoreNLPParser(url='http://0.0.0.0:9000', tagtype='pos')
lemmatizer = WordNetLemmatizer()


def extract_triplet(input_sent, output=['parse_tree']):
    # Parse the input sentence with Stanford CoreNLP Parser
    parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
    # Extract subject, predicate and object
    subjects = extract_subject(parse_tree)
    predicates = extract_predicate(parse_tree)
    objects = extract_object(parse_tree)
    if 'parse_tree' in output:
        print('---Parse Tree---')
        parse_tree.pretty_print()
    if 'spo' in output:
        print('---Subject---')
        print(subjects)
        print('---Predicate---')
        print(predicates)
        print('---Object---')
        print(objects)
    subject = lemmatizer.lemmatize(subjects[0])
    predicate = lemmatizer.lemmatize(predicates[0], 'v')
    object = lemmatizer.lemmatize(objects[0])

    if 'result' in output:
        print('---Result---')
        print(' '.join([subject, predicate, object]))
    return subject.replace(' ', '_'), predicate, object.replace(' ', '_')


def extract_subject(parse_tree):
    subject = []
    leaves = list(parse_tree.subtrees(lambda y: y.height() == 2))
    j = 0

    for i, leaf in enumerate(leaves):
        if i < j:
            continue
        if leaf.label() in ['JJ', 'NN']:
            temp = leaf[0]
            j = i + 1
            while j < len(leaves) and leaves[j].label() == 'NN':
                temp += (' ' + leaves[j][0])
                j += 1
            subject.append(temp)
    if len(subject) != 0:
        return [subject[0]]
    else:
        return ['']


def extract_predicate(parse_tree):
    output, predicate = [], []
    leaves = list(parse_tree.subtrees(lambda y: y.height() == 2))
    j = 0

    for i, leaf in enumerate(leaves):
        if i < j:
            continue
        if leaf.label() in ['IN', 'TO', 'JJR'] or leaf.label().startswith('VB'):
            if leaf.label() in ['IN', 'TO', 'JJR']:
                temp = leaf[0]
                if leaf.label() == 'JJR' and i + 1 < len(leaves) and leaves[i + 1].label() == 'IN':
                    temp += (' ' + leaves[i + 1][0])
                    j = i + 2
                elif i + 2 < len(leaves) and leaves[i + 2].label() in ['IN', 'TO']:
                    temp += (' ' + leaves[i + 1][0] + ' ' + leaves[i + 2][0])
                    print(temp)
                    j = i + 3
                predicate.append(temp)
            else:
                if leaf[0] in ['is', 'was', 'were']:
                    temp = ''
                else:
                    temp = leaf[0]

                if i + 1 < len(leaves) and leaves[i + 1].label() in ['IN', 'TO', 'JJR']:
                    temp += (' ' + leaves[i + 1][0])
                    if i + 2 < len(leaves) and leaves[i + 1].label() == 'JJR' and leaves[
                        i + 2].label() == 'IN':
                        temp += (' ' + leaves[i + 2][0])
                        j = i + 3
                    elif i + 3 < len(leaves) and leaves[i + 3].label() in ['IN', 'TO']:
                        temp += (' ' + leaves[i + 2][0] + ' ' + leaves[i + 3][0])
                        print(temp)
                        j = i + 4
                    else:
                        j = i + 2
                predicate.append(temp.strip())
    if len(predicate) != 0:
        return [predicate[-1]]
    else:
        return ['']


def extract_object(parse_tree):
    objects, output, word = [], [], []
    leaves = list(parse_tree.subtrees(lambda y: y.height() == 2))
    j = 0

    for i, leaf in enumerate(leaves):
        if i < j:
            continue
        if leaf.label() in ['JJ', 'NN']:
            temp = leaf[0]
            j = i + 1
            while j < len(leaves) and leaves[j].label() == 'NN':
                temp += (' ' + leaves[j][0])
                j += 1
            objects.append(temp)
    if len(objects) >= 2:
        return [objects[-1]]
    else:
        return ['']


if __name__ == '__main__':
    extract_triplet('person rides horse', output=['parse_tree'])
