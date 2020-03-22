import scispacy
import spacy
import json
from spacy.vocab import Vocab
import numpy as np
from tqdm import tqdm
from config import *
import concurrent.futures
import os
import gensim
import time
from dataset import PretrainedSpacyDataset
import pickle

OWN_VECTORS = os.path.join(DATASET_PROCESSED, 'own_vectors.pickle')
dataset = PretrainedSpacyDataset(dataset_path=DATASET_PICKLE, vectors_dataset_path=OWN_VECTORS, num_dimensions=200, mode='raw')

"""
    PREPROCESSING
"""
def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)

if not os.path.exists(".cache.texts"):
    nlp = spacy.load('en_core_sci_lg', disable=['ner', 'parser'])

    t = time.time()
    texts = []
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        papers = dataset.documents[i:i+batch_size]
        for doc in nlp.pipe([paper.text for paper in papers], batch_size=batch_size, n_threads=-1):
            aux = cleaning(doc)
            if aux is not None:
                texts.append(aux)
        print(i, len(dataset.documents))

    print('Time to clean up everything: {} mins'.format(round((time.time() - t) / 60, 2)))
    with open(".cache.texts", 'wb') as handle:
        pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Save texts in cache')
else:
    print('Load texts from cache')
    with open(".cache.texts", 'rb') as handle:
        texts = pickle.load(handle)


"""
    TRAINING WORD2VEC
"""
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

model = gensim.models.Word2Vec(
    texts,
    size=200,
    window=10,
    min_count=2,
    workers=os.cpu_count(),
    iter=100,
    compute_loss=True, 
    callbacks=[callback()])
model.save("word2vec.model")