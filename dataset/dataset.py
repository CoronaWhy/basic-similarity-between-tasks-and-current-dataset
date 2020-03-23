import faiss
import numpy as np
import pickle
from scipy.spatial import distance
import concurrent.futures
from threading import Thread
from functools import partial
from collections import defaultdict, OrderedDict

import schedule
import time
import kaggle
import os
import glob
import glob2
import json
from tqdm import tqdm

class Document:
    """
    ==============================================================================
        CONSTRUCTOR
    ==============================================================================
    """
    def __init__(self, dataset, raw_dict, mean_vector_dict=None):
        self.dataset = dataset
        self.raw_dict = raw_dict
        self.mean_vector_dict = mean_vector_dict

        if mean_vector_dict is None and self.dataset.compute_vectors:
            self.create_vectors()

    @property
    def id(self):
        return self.raw_dict['_id']

    @property
    def title(self):
        return self.raw_dict['title']

    @property
    def abstract(self):
        return self.raw_dict['sections']['abstract']

    @property
    def text(self):
        return '\n'.join([self.raw_dict['sections'][section] for section in self.raw_dict['sections_order']])

    @property
    def full(self):
        return self.text
    

    @property
    def body(self):
        return '\n'.join([self.raw_dict['sections'][section] for section in filter(lambda x: x != 'abstract', self.raw_dict['sections_order'])])
    

    """
    ==============================================================================
        GET VECTORS
    ==============================================================================
    """
    def create_title_mean(self):
        if self.mean_vector_dict is None:
            self.mean_vector_dict = {}

        if 'title' not in self.mean_vector_dict:
            vector, num_words = self.dataset.get_mean_vector(self.raw_dict['title'])
            self.mean_vector_dict['title'] = {'vector': vector, 'num_words': num_words}

    def mean_vector_title(self):
        if self.mean_vector_dict is None:
            self.create_document_mean()

        return self.mean_vector_dict['title']['vector']

    def create_document_mean(self):
        if self.mean_vector_dict is None:
            self.mean_vector_dict = {}

        if 'sections' not in self.mean_vector_dict:
            self.mean_vector_dict['sections'] = {}
            for section, section_text in self.raw_dict['sections'].items():
                vector, num_words = self.dataset.get_mean_vector(section_text)
                self.mean_vector_dict['sections'][section] = {'vector': vector, 'num_words': num_words}

    def mean_vector(self, section=None):
        if self.mean_vector_dict is None:
            self.create_document_mean()

        if section is None:
            mean = np.zeros(shape=(self.dataset.num_dimensions, ))
            den = 0
            for section, vector_block in self.mean_vector_dict['sections'].items():
                vector = vector_block['vector']
                num_words = vector_block['num_words']

                mean += vector * num_words
                den += num_words
            
            if den == 0:
                return mean
            
            return mean / den

        if section == 'body':
            mean = np.zeros(shape=(self.dataset.num_dimensions, ))
            den = 0
            for section, vector_block in self.mean_vector_dict['sections'].items():
                if section == 'abstract':
                    continue
                
                vector = vector_block['vector']
                num_words = vector_block['num_words']

                mean += vector * num_words
                den += num_words
            
            if den == 0:
                return mean
            
            return mean / den

        return self.mean_vector_dict['sections'][section]['vector']

    def create_vectors(self):
        self.create_title_mean()
        self.create_document_mean()

    # Numpy array of the all valid vectors: (N, EMBEDDING_SIZE). N num valid words
    def vector_title_array(self):
        return self.dataset.get_vectors(self.raw_dict['title'])

    def vector_array(self, section=None):
        if section is None:
            return np.stack([self.dataset.get_vectors(self.raw_dict['sections'][section]) \
                for section in self.raw_dict['sections_order']], axis=0)  
        
        elif section == 'body':
            return np.stack([self.dataset.get_vectors(self.raw_dict['sections'][section]) \
                for section in filter(lambda x: x != 'abstract', self.raw_dict['sections_order'])], axis=0)   

        return self.dataset.get_vectors(self.raw_dict['sections'][section])

    """
    ==============================================================================
        UTILS
    ==============================================================================
    """
    def __repr__(self):
        return self.raw_dict['title']

class Dataset:
    TITLE, ABSTRACT, SECTION, BODY = range(4)

    """
    ==============================================================================
        UTILS
    ==============================================================================
    """
    def get_mean_vector(self, text):
        raise NotImplemented

    def get_vectors(self, text):
        raise NotImplemented

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)
    

    """
    ==============================================================================
        CONSTRUCTOR
    ==============================================================================
    """
    def add_raw_document(self, paper, mean_vectors=None):
        doc = Document(self, paper, mean_vectors)
        self.hash_to_idx[doc.id] = len(self.documents)
        self.documents.append(doc)

    def add_raw_document_from_tuple(self, data):
        self.add_raw_document(paper=data[0], mean_vectors=data[1])

    def parse_dataset(self, dataset, vectors_dataset=None):
        if vectors_dataset is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                tuple_data = list(zip(dataset, vectors_dataset))
                list(tqdm(executor.map(self.add_raw_document_from_tuple, tuple_data), total=len(tuple_data)))
                
        else: 
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                list(tqdm(executor.map(self.add_raw_document, dataset), total=len(dataset)))

    def __init__(self, num_dimensions, dataset_path=None, vectors_dataset_path=None, num_centroids=1, similarity_metric='cosine', mode='compare'):
        self.dataset_path = dataset_path
        self.vectors_dataset_path = vectors_dataset_path

        if mode.lower() == 'compare':
            self.compute_vectors = True
            self.create_indexes = True
        
        elif mode.lower() == 'integrate':
            self.compute_vectors = True
            self.create_indexes = False
        
        elif mode.lower() == 'raw':
            self.compute_vectors = False
            self.create_indexes = False
        
        self.similarity_metric = similarity_metric
        self.num_dimensions = num_dimensions
        self.num_centroids = num_centroids

        # Hash to indices
        self.hash_to_idx = {}
        # Documents List
        self.documents = []

        if self.dataset_path is not None and os.path.exists(self.dataset_path):
            print('PARSING DATASET...', end=' ')
            with open(self.dataset_path, 'rb') as handle:
                dataset = pickle.load(handle)

            if self.vectors_dataset_path is not None and os.path.exists(self.vectors_dataset_path):
                with open(self.vectors_dataset_path, 'rb') as handle:
                    vectors_dataset = pickle.load(handle)
            else:
                vectors_dataset = None

            self.parse_dataset(dataset, vectors_dataset)
            print('DONE')

        if self.compute_vectors and self.create_indexes:
            print('FAISS INDEXING...', end=' ')
            self.create_faiss()
            print('DONE')

    """
    ==============================================================================
        SEARCHING
    ==============================================================================
    """
    def create_faiss(self):
        quantiser = faiss.IndexFlatL2(self.num_dimensions)
        self.faiss_params = {}
        if self.similarity_metric == 'cosine':
            self.faiss_params['preprocess_opt'] = 'norm'
            self.faiss_params['metric'] = faiss.METRIC_L2
        
        elif self.similarity_metric == 'inner':
            self.faiss_params['preprocess_opt'] = 'false'
            self.faiss_params['metric'] = faiss.METRIC_INNER_PRODUCT
        
        elif self.similarity_metric == 'euclidean':
            self.faiss_params['preprocess_opt'] = 'false'
            self.faiss_params['metric'] = faiss.METRIC_L2

        elif self.similarity_metric == 'mahalanobis':
            self.faiss_params['preprocess_opt'] = 'covar'
            self.faiss_params['metric'] = faiss.METRIC_L2

        self.faiss_indices = {
            'title': faiss.IndexIVFFlat(quantiser, self.num_dimensions, self.num_centroids, self.faiss_params['metric']),
            'abstract': faiss.IndexIVFFlat(quantiser, self.num_dimensions, self.num_centroids, self.faiss_params['metric']),
            'body': faiss.IndexIVFFlat(quantiser, self.num_dimensions, self.num_centroids, self.faiss_params['metric']),
        }

        # Title train
        vectors = self.search_preprocess(np.stack([doc.mean_vector_title() for doc in self.documents], axis=0), is_train=True)
        self.faiss_indices['title'].train(vectors)
        self.faiss_indices['title'].add(vectors)

        # Train abstract
        vectors = self.search_preprocess(np.stack([doc.mean_vector('abstract') for doc in self.documents], axis=0), is_train=True)
        self.faiss_indices['abstract'].train(vectors)
        self.faiss_indices['abstract'].add(vectors)

        # Train full
        vectors = self.search_preprocess(np.stack([doc.mean_vector('body') for doc in self.documents], axis=0), is_train=True)
        self.faiss_indices['body'].train(vectors)
        self.faiss_indices['body'].add(vectors)

    def search_preprocess(self, data, is_train=False):
        if self.faiss_params['preprocess_opt'] == 'norm':
            return np.float32((data + 1e-6) / (np.linalg.norm(data + 1e-6, keepdims=True, axis=-1) + 1e-30))
        elif self.faiss_params['preprocess_opt'] == 'false':
            return np.float32(data)
        elif self.faiss_params['preprocess_opt'] == 'covar':
            if is_train:
                cov = np.np.cov(data)
                L = np.linalg.cholesky(cov)
                self.faiss_params['mahalanobis_transform'] = np.linalg.inv(L)
            return np.float32(np.dot(data, self.faiss_params['mahalanobis_transform'].T))

    def get_similar_docs_than(self, data, k=10, by=None, section=None):
        if not self.compute_vectors:
            raise IndexError('Compute vector attribute is disabled')

        if by == None:
            by = Dataset.BODY

        # Data is a document
        if isinstance(data, Document):
            if by == Dataset.TITLE:
                vector = data.mean_vector_title()
            
            elif by == Dataset.ABSTRACT:
                vector = data.mean_vector()
            
            elif by == Dataset.SECTION:
                raise NotImplemented
                #assert(section is not None)
                #vector = data.mean_vector(section)
            
            elif by == Dataset.BODY:
                vector = data.mean_vector('body')

        # Data is a string
        elif isinstance(data, str):
            vector, _ = self.get_mean_vector(data)

        vector = self.search_preprocess(vector)

        # Find similars
        if by == Dataset.TITLE:
            _, indices = self.faiss_indices['title'].search(np.expand_dims(vector, axis=0), k)
        
        elif by == Dataset.ABSTRACT:
            _, indices = self.faiss_indices['abstract'].search(np.expand_dims(vector, axis=0), k)
        
        elif by == Dataset.SECTION:
            raise NotImplemented
        
        elif by == Dataset.BODY:
            _, indices = self.faiss_indices['body'].search(np.expand_dims(vector, axis=0), k)

        """if by == Dataset.TITLE:
            aux_vectors = np.stack([doc.mean_vector_title() for doc in self.documents], axis=0)
        
        elif by == Dataset.ABSTRACT:
            aux_vectors = np.stack([doc.mean_vector('abstract') for doc in self.documents], axis=0)
        
        elif by == Dataset.SECTION:
            raise NotImplemented
        
        elif by == Dataset.BODY:
            aux_vectors = np.stack([doc.mean_vector('body') for doc in self.documents], axis=0)

        distances = distance.cdist(np.expand_dims(vector, axis=0), aux_vectors, "cosine")
        indices = distances.argsort(axis=-1)[:, :k]"""

        return [self.documents[idx] for idx in indices[0]]

    """
    ==============================================================================
        HASH ID UTILS
    ==============================================================================
    """
    def exists(self, hash_id):
        return hash_id in self.hash_to_idx.keys()

    def get_by_id(self, hash_id):
        return self.documents[self.hash_to_idx[hash_id]]

    """
    ==============================================================================
        SCAN AND SAVE
    ==============================================================================
    """
    def parse_document_json(self, json_path):
        data = {}
        with open(json_path) as json_file:
            json_data = json.load(json_file)

            data['_id'] = json_data['paper_id']
            
            if self.exists(data['_id']):
                return None

            data['title'] = json_data['metadata']['title']
            data['authors'] = json_data['metadata']['authors']
            data['bib_entries'] = json_data['bib_entries']
            data['ref_entries'] = json_data['ref_entries']

            data['citations'] = defaultdict(list)
            data['sections'] = defaultdict(lambda: "")

            # Abstract
            if isinstance(json_data['abstract'], (list, tuple)):
                try:
                    data['sections']['abstract'] = json_data['abstract'][0]['text']
                    data['citations']['abstract'] += [{'start': cite['start'], 'end': cite['end'], 'ref_id': cite['ref_id']} for cite in json_data['abstract'][0]['cite_spans']]
                except:
                    data['sections']['abstract'] = ''
            else:
                data['sections']['abstract'] = json_data['abstract']

            offsets = defaultdict(lambda: 0)
            sections_order = OrderedDict()
            for block_text in json_data['body_text']:
                text = block_text['text']
                section = block_text['section']
                data['sections'][section] += text
                data['citations'][section] += [{'start': offsets[section] + cite['start'], 'end': offsets[section] + cite['end'], 'ref_id': cite['ref_id']} for cite in block_text['cite_spans']]
                offsets[section] += len(text)

                if section not in sections_order:
                    sections_order[section] = True
            
            data['sections_order'] = list(sections_order.keys())
            data['sections'] = dict(data['sections'])
            data['citations'] = dict(data['citations'])
        
        return data

    def scan_file(self, json_path):
        raw_doc = self.parse_document_json(json_path)
        if raw_doc is not None:
            self.add_raw_document(raw_doc)

    def scan_folder(self, folder_path):
        for folder_path in filter(lambda folder_path: os.path.isdir(folder_path), glob2.iglob(os.path.join(folder_path, "*"))):
            folder_name = os.path.basename(folder_path)
            print('\tProcessing %s folder' % (folder_name, ))
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                list_jsons = glob2.glob(os.path.join(folder_path, "**", "*.json"))
                list(tqdm(executor.map(self.scan_file, list_jsons), total=len(list_jsons)))

    def save(self):
        raw_dicts = []
        vector_dicts = []
        for doc in self.documents:
            raw_dicts.append(doc.raw_dict)
            vector_dicts.append(doc.mean_vector_dict)

        with open(self.dataset_path, 'wb') as handle:
            pickle.dump(raw_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.vectors_dataset_path, 'wb') as handle:
            pickle.dump(vector_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    ==============================================================================
        SYNC: CONTINOUS INTEGRATION
    ==============================================================================
    """
    def sync(self, dataset_name, folder_path, callback=None):
        def __sync_thread():
            print('Checking new changes...')
            
            # Download from kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset_name, path=folder_path, unzip=True)

            # Create new dataset with the changes
            self.scan_folder(folder_path)
            self.save()

            # Execute callback
            if callback is not None:
                callback(self)
        
        t = Thread(target=__sync_thread)
        t.start()

        schedule.every().hour.do(__sync_thread)
        while True:
            schedule.run_pending()
            time.sleep(3600)

class PretrainedSpacyDataset(Dataset):
    def __init__(self, num_dimensions, dataset_path=None, vectors_dataset_path=None, num_centroids=1, similarity_metric='cosine', mode='compare'):
        import scispacy
        import spacy
        self.nlp = spacy.load("en_core_sci_lg")

        super().__init__(num_dimensions=num_dimensions, 
            dataset_path=dataset_path, 
            vectors_dataset_path=vectors_dataset_path, 
            num_centroids=num_centroids, 
            similarity_metric=similarity_metric,
            mode=mode)  

    def get_mean_vector(self, text):
        tokens = self.nlp(text.lower())
        return tokens.vector, sum([token.has_vector for token in tokens])

    def get_vectors(self, text):
        tokens = self.nlp(text.lower())
        vectors = []
        for token in tokens:
            if token.has_vector:
                vectors.append(token.vector)

        if len(vectors) > 0:
            vectors = np.stack(vectors, axis=0)
        else:
            vectors = np.zeros(shape=(0, self.num_dimensions))

        return vectors

class PaperWord2VecDataset(Dataset):
    def __init__(self, num_dimensions, dataset_path=None, vectors_dataset_path=None, num_centroids=1, similarity_metric='cosine', mode='compare'):
        import scispacy
        import spacy
        from gensim.models import KeyedVectors
        self.nlp = spacy.load("en_core_sci_lg")
        self.wv = KeyedVectors.load(os.path.join("models", "word2vec", "word2vec_using_papers.bin"))

        super().__init__(num_dimensions=num_dimensions, 
            dataset_path=dataset_path, 
            vectors_dataset_path=vectors_dataset_path, 
            num_centroids=num_centroids, 
            similarity_metric=similarity_metric,
            mode=mode)  

    def get_mean_vector(self, text):
        vectors = self.get_vectors(text)
        if np.prod(vectors.shape) == 0:
            return np.zeros(shape=(0, self.num_dimensions)), 0
        return np.mean(vectors, axis=0), vectors.shape[0]

    def get_vectors(self, text):
        vectors = []
        for token in self.nlp(text.lower()):
            if not token.is_stop:
                try:
                    vector = self.wv[token.lemma_]
                    vectors.append(vector)
                except:
                    pass

        if len(vectors) > 0:
            vectors = np.stack(vectors, axis=0)
        else:
            vectors = np.zeros(shape=(0, self.num_dimensions))

        return vectors

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import *

    dataset = PretrainedSpacyDataset(dataset_path=DATASET_PICKLE, vectors_dataset_path=SCAPY_PRETRAINED_VECTORS_DATASET_PICKLE, num_dimensions=200, mode='compare')
    tasks = LOAD_TASKS()

    for i in range(len(tasks)):
        print(tasks[i]["title"] + "\n")
        documents = dataset.get_similar_docs_than(tasks[i]['description'], k=10)
        for j, doc in enumerate(documents):
            print(f"Rank {j+1}: \nPaper ID: {doc.id} \n" + doc.text[:500] + "\n")

    # v1 = dataset.get_by_id('1378320afa873bdb81e3f3314a430c7a208d2d08').mean_vector('body')
    # v2 = dataset.get_mean_vector(tasks[0]['description'])
    # print(distance.cdist(np.expand_dims(v1, axis=0), np.expand_dims(v2, axis=0), "cosine"))