from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import gensim
from gensim.models.callbacks import CallbackAny2Vec
import os
import glob

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
        if self.epoch % 10 == 0:
            output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
            model.save(os.path.join(self.folder_path, output_path))
            self.epoch += 1

model = gensim.models.Word2Vec.load(max(glob.glob('checkpoints/*.model'), key=os.path.getctime))
model.wv.save("./result/word2vec_using_papers.bin")