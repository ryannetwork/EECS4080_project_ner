import warnings
import spacy
import argparse
import os

warnings.filterwarnings("ignore")
from collections import Counter
import sys
import conllToSpacy
import time
from spacy.tokens import Span
from test_ner_spacy1 import compute_score
from sklearn.metrics import accuracy_score
import string

# Parsing argument for command-line.
parser = argparse.ArgumentParser(description="Testing an NER model with SpaCy.")
parser.add_argument("-tp", "--test_path", help="Path to CoNLL test dataset.")
parser.add_argument("-model", help="Path to the model.")
args = parser.parse_args()

if args.test_path:
    testing, ent_true = conllToSpacy.main(args.test_path, test='test')
    # testing = testing[:20]
    # ent_true = ent_true[:20]
    text = " ".join(testing)
    print("Got test data at", args.test_path)
else:
    print("No test data path given ! Interrupting the script...")
    sys.exit()
if args.model:
    model = args.model
    print("Model loaded at", model)
else:
    print("No model path given !")
    sys.exit()


def ner_post_pro(model):
    nlp2 = spacy.load(model)
    start = time.time()
    doc = nlp2(text)

    # add all the ent strings into a set
    print("remove repeat ent...")
    ent_set = set()
    # remove repeat ent and ent that are not noun
    # add all the ent strings that are noun into a dict as keys, ent as values
    ent_Noun_dict = {}
    for ent in doc.ents:
        if ent.text not in string.punctuation:
            ent_set.add(ent.text)
            if ent.text not in ent_Noun_dict:
                ent_Noun_dict[ent.text] = ent
    print(len(ent_Noun_dict))
    print(ent_Noun_dict)

    # add ent pairs to similarity_dict
    # calculate the similarity between each ent pairs
    print("add ent pairs to similarity_dict...")
    count = 0
    similarity_dict = {}
    for k1 in ent_Noun_dict.keys():
        count += 1
        print(k1, count)
        for k2 in ent_Noun_dict.keys():
            s = ent_Noun_dict[k1].similarity(ent_Noun_dict[k2])
            if s > 0.6:  # threshold
                if k1 not in similarity_dict:
                    similarity_dict[k1] = [ent_Noun_dict[k2].label_]
                else:
                    similarity_dict[k1].append(ent_Noun_dict[k2].label_)

    # find the most frequent label in the list of labels
    print("find most common label...")
    label_dict_update = {}
    for k in similarity_dict:
        label_dict_update[k] = Counter(similarity_dict[k]).most_common(1)[0][0]

    # update the ent list
    print("update ent list...")
    label_dict = label_dict_update
    change_list = []
    change_dict = {}
    for ent in doc.ents:
        # if ent is in label_dict and its label is different from label in the dict, we need to update it
        if ent.text in label_dict and ent.label_ != label_dict[ent.text]:
            # remove the ent from the ent list
            ents = list(doc.ents)
            ents.remove(ent)
            doc.ents = tuple(ents)
            # add ent to the change_list
            change_list.append(ent)
            if ent.text not in change_dict:
                change_dict[ent.text] = [ent.label_, label_dict[ent.text]]

    update_ent = []
    for e in change_list:
        span = Span(doc, e.start, e.end, label=label_dict[e.text])
        update_ent.append(span)
    print(change_dict)
    doc.ents = list(doc.ents) + update_ent
    print("time", time.time() - start)
    return doc


def test_after_post_pro(doc):
    ent_true_group = []
    for ele in ent_true:
        ent_true_group = ent_true_group + ele
    ent_true_group = [ent_true_group]
    ent_pred = []
    entities = ['O'] * len(text.split())

    for ent in doc.ents:
        try:
            entities[text.split().index(ent.text)] = ent.label_
        except IndexError:
            print('Index Error! Ent:', list(ent.text), '. Text:', text)
        except ValueError:
            print('Value Error! Ent:', list(ent.text), '. Text:', text)
    ent_pred.append(entities)

    print('Pred Entity: ', set([x for y in ent_pred for x in y]))
    print('True Entity: ', set([x for y in ent_true_group for x in y]))
    y_pred = [x for y in ent_pred for x in y]
    y_true = [x for y in ent_true_group for x in y]

    Precision = compute_score(y_pred, y_true)
    Recall = compute_score(y_true, y_pred)
    F1_score = 2 * (Recall * Precision) / (Recall + Precision)
    print("Random accuracy: %0.2f" % (accuracy_score(y_true, ['O'] * len(y_true))))
    print("Accuracy score: %0.2f" % (accuracy_score(y_true, y_pred)))
    print('Precision: %0.2f' % (Precision))
    print('Recall: %0.2f' % (Recall))
    print('F1 score: %0.2f' % (F1_score))
    with open('score.csv', 'a', encoding='utf-8') as f:
        f.write(os.path.basename(model) + ',' + str(Precision) + ',' + str(Recall) + ',' + str(F1_score) + '\n')


if __name__ == "__main__":
    doc = ner_post_pro(model)
    test_after_post_pro(doc)
