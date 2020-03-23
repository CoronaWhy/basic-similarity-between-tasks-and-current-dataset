import os
import pickle
import json

def __LOAD(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


"""
    MAIN PATHS
"""
UPDATE_DATASET = False
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURR_PATH, 'dataset')
os.makedirs(DATASET_PATH, exist_ok=True)

RAW_DATASET_PATH = os.path.join(DATASET_PATH, 'raw')
os.makedirs(RAW_DATASET_PATH, exist_ok=True)

DATASET_PROCESSED = os.path.join(DATASET_PATH, 'processed')
os.makedirs(DATASET_PROCESSED, exist_ok=True)

DATASET_PICKLE = os.path.join(DATASET_PROCESSED, 'dataset.pickle')

TASKS_JSON = os.path.join(DATASET_PATH, 'tasks.json')
def LOAD_TASKS():
    global TASKS_JSON
    with open(TASKS_JSON) as json_file:
        return json.load(json_file) 

"""
    WORD2VEC PATHS
"""
SCAPY_PRETRAINED_VECTORS_DATASET_PICKLE = os.path.join(DATASET_PROCESSED, 'vectors_dataset.pickle')

WORD2VEC_VECTORS_DATASET_PICKLE = os.path.join(DATASET_PROCESSED, 'word2vec_dataset.pickle')

CORONAVIRUS_SPECIFIC_VOCAB_PATH = os.path.join(DATASET_PROCESSED, 'corona_vectors')