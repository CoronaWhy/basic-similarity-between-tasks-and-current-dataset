from collections import defaultdict, OrderedDict
from config import *
import pickle

import scispacy
import spacy

import glob
import glob2
from dataset import PretrainedSpacyDataset, PaperWord2VecDataset

"""
==============================================================================
    DOWNLOADING FROM KAGGLE
==============================================================================
"""
print('0 - DOWNLOADING LAST DATASET FROM KAGGLE')
#dataset = PretrainedSpacyDataset(dataset_path=DATASET_PICKLE, vectors_dataset_path=SCAPY_PRETRAINED_VECTORS_DATASET_PICKLE, num_dimensions=200, mode='integrate')
#dataset.sync('allen-institute-for-ai/CORD-19-research-challenge', RAW_DATASET_PATH)

dataset = PaperWord2VecDataset(dataset_path=DATASET_PICKLE, vectors_dataset_path=WORD2VEC_VECTORS_DATASET_PICKLE, num_dimensions=200, mode='integrate')
dataset.sync('allen-institute-for-ai/CORD-19-research-challenge', RAW_DATASET_PATH)
print('¡DONE!')
print()