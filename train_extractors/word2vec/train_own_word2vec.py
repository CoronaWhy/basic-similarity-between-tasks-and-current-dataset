import scispacy
import spacy
import json
from spacy.vocab import Vocab
import numpy as np
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *
import concurrent.futures
import os
import gensim
import time
from dataset import PretrainedSpacyDataset
import pickle
from gensim.models.callbacks import CallbackAny2Vec


#OWN_VECTORS = os.path.join(DATASET_PROCESSED, 'own_vectors.pickle')
dataset = PretrainedSpacyDataset(dataset_path=DATASET_PICKLE, vectors_dataset_path=None, num_dimensions=200, mode='raw')

"""
    PREPROCESSING
"""
def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return txt

if not os.path.exists("../.cache.texts"):
    nlp = spacy.load('en_core_sci_lg', disable=['ner', 'parser'])

    t = time.time()
    texts = []
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        papers = dataset.documents[i:i+batch_size]
        for doc in nlp.pipe([paper.text for paper in papers], batch_size=batch_size, n_threads=-1):
            sentence = cleaning(doc)
            if sentence is not None:
                texts.append(sentence)
        print(i, len(dataset.documents))

    print('Time to clean up everything: {} mins'.format(round((time.time() - t) / 60, 2)))
    with open("../.cache.texts", 'wb') as handle:
        pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Save texts in cache')
else:
    print('Load texts from cache')
    with open("../.cache.texts", 'rb') as handle:
        texts = pickle.load(handle)


class ModelLoss(CallbackAny2Vec):
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
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, folder_path, path_prefix):
        self.folder_path = folder_path
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        model.save(os.path.join(self.folder_path, output_path))
        self.epoch += 1

"""
    TRAINING WORD2VEC
"""

print('Word2Vec mode')
model = gensim.models.Word2Vec(
    texts,
    size=200,
    window=10,
    min_count=2,
    workers=os.cpu_count() // 2,
    iter=80,
    compute_loss=True, 
    callbacks=[ModelLoss(), EpochSaver("checkpoints", "word2vec")])
model.save(os.path.join("checkpoints", "word2vec.1000.model"))
