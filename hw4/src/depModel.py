import os,sys
from decoder import *
from train import *
import tensorflow as tf
import numpy as np
from collections import defaultdict

class DepModel:
    def __init__(self):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        # write your code here for additional parameters.
        # feel free to add more arguments to the initializer.

        ## Load training vocabs
        vocab = Vocab(vocabs_dir)
        vocab.get_dic()
        ## get the actions list
        self.actions = []
        for ac in vocab.get_actions():
            self.actions.append(ac[0])
        self.sess = tf.Session()
        self.model = Model(vocab, mode = 'test')
        model_id = sys.argv[4]
        model_path = os.path.join('./models', sys.argv[3], 'check_points', 'model.ckpt-{}'.format(model_id))
        self.model.load(model_path, self.sess)

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # change this part of the code.
        scores = self.model.predict(self.sess, str_features)
        scores = np.reshape(scores,[len(self.actions)])
        scores_list = scores.tolist()
        return scores_list

if __name__=='__main__':
    m = DepModel()
    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(m.score, m.actions).parse(input_p, output_p)
