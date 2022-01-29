# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


def readdata():
    with open("./datasets/Training/node_features_text.json") as jf:
        data = json.load(jf)

    input = [None] * len(data)

    maxint = 0
    for k in range(len(input)):
        input[k] = data[str(k)]
        maxint = max(maxint, max(input[k]))
    inputarray = np.zeros([len(input), maxint+1])
    inputsent = []



    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(input)]

    ## Train doc2vec model
    model = Doc2Vec(tagged_data, vector_size=50, window=2, min_count=1, workers=4, epochs=100)
    # Save trained doc2vec model
    model.save("test_doc2vec_50.model")

    ## Load saved doc2vec model
    model = Doc2Vec.load("test_doc2vec_50.model")
    ## Print model vocabulary
    print(model.docvecs[0])
    print(model.docvecs[1])
    bp = 0

readdata()