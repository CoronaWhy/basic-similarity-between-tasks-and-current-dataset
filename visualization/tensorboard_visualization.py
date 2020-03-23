import numpy as np
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from dataset import PretrainedSpacyDataset
dataset = PretrainedSpacyDataset(dataset_path=DATASET_PICKLE, vectors_dataset_path=VECTORS_DATASET_PICKLE, num_dimensions=200, mode='integrate')

writer = SummaryWriter('results')

vectors = []
metadata = []
for i, doc in enumerate(dataset):
	vector = doc.mean_vector()
	if np.prod(vector.shape) == 0:
		continue

	vectors.append(vector)
	if isinstance(doc.title, str):
		metadata.append(doc.id + " || " + doc.title.encode("ascii","ignore").decode("ascii"))
	else:
		metadata.append('undefined')

	if i % 100 == 0:
		print(i, len(dataset))

vectors = np.stack(vectors, axis=0)

writer.add_embedding(vectors, metadata)
writer.close()