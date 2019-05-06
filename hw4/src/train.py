__author__="Yue Luo <yl4003@columbia.edu>"
__date__ ="$May 1, 2019"

import os
from collections import defaultdict
import tensorflow as tf
import numpy as np
import random

# Print Logs or not
verbose = True
# Hardcode paths used for training
train_path = './data/train.data'
vocabs_dir = './data'
model_dir = './models/q1'  ## change the last one for different dirs in each question (q1/q2/q3)
# Hardcode parameters used in the model
batchsize = 2000
lr = 0.01
epoch = 10
word_em_dim = 128
pos_em_dim = 64
label_em_dim = 64
h1_dim = 400   ## Change these two for different settings in each question. For part1:200, part2:400
h2_dim = 400
## For training print and model saving
show_iter = 10


def file_to_dic(path):
    """
    convert a path into a dic
    """
    res = defaultdict(int)
    file = open(path, 'r')
    l = file.readline()
    while l:
        line = l.strip()
        fields = line.split(" ")
        res[fields[0]] = int(fields[1])
        l = file.readline()
    return res

class Vocab:
    """
    A class to store all training vocaulary.
    """
    def __init__(self, vocabs_dir):
        self.vocabs_dir = vocabs_dir
        self.vocabs_label_path = os.path.join(vocabs_dir, 'vocabs.labels')
        self.vocabs_word_path = os.path.join(vocabs_dir, 'vocabs.word')
        self.vocabs_pos_path = os.path.join(vocabs_dir, 'vocabs.pos')
        self.vocabs_action_path = os.path.join(vocabs_dir, 'vocabs.actions')
        self.label = defaultdict(int)
        self.word = defaultdict(int)
        self.pos = defaultdict(int)
        self.action= defaultdict(int)

    def get_actions(self):
        """
        Get the actions list for prediction prediction purpose
        Sort the action list by values
        """
        sort_action = sorted(self.action.items(), key=lambda (k,v): v, reverse=False)
        return sort_action

    def get_dic(self):
        """
        get the dic. All dics are with key=#string and value=index.
        """
        print "Getting vocabs..."
        self.label = file_to_dic(self.vocabs_label_path)
        self.word = file_to_dic(self.vocabs_word_path)
        self.pos = file_to_dic(self.vocabs_pos_path)
        self.action = file_to_dic(self.vocabs_action_path)
        self.n_w = len(self.word)
        self.n_t = len(self.pos)
        self.n_l = len(self.label)
        self.n_a = len(self.action)
        if verbose:
            self.check_len()

    def check_len(self):
        print "Length of Vocabs:"
        print "  Words:%i" % len(self.word)
        print "  POS:%i" % len(self.pos)
        print "  Labels:%i" % len(self.label)
        print "  Actions:%i" % len(self.action)

    def get_size(self):
        """
        Get the size of the vocabs.
        """
        return self.n_w, self.n_t, self.n_l, self.n_a

    def get_ind(self, key, case):
        if case == 'w': ## For word not exist, will return 0 -> the val of <unk>
            if key != '<unk>' and self.word[key] == 0: ## For non exist POS tag, use <null>
                return self.word['<unk>']
            return self.word[key]
        elif case == 't':
            if key != 'PRP$' and self.pos[key] == 0: ## For non exist POS tag, use <null>
                return self.pos['<null>']
            return self.pos[key]
        elif case == 'l':
            if key != 'rroot' and self.label[key] == 0: ## For non exist label, use <null>
                return self.label['<null>']
            return self.label[key]
        elif case == 'a':
            return self.action[key]
        else:
            raise ValueError('Please check the query case...')
            exit()

    def parse_input(self, input, n_w, n_t, n_l):
        """
        Use to parse a single line of input. Use for inference purpose.
        Input is a list of len=52(no action). Some values may not exist in the vocab.
        return numpy arrays for word, pos and label.
        """
        assert len(input) == 52
        word_ = input[:20]
        pos_ = input[20:40]
        label_ = input[40:52]

        word_np = np.zeros((20, n_w))
        pos_np = np.zeros((20, n_t))
        label_np = np.zeros((12, n_l))

        # Process word
        for w_i, w in enumerate(word_):
            w_em = self.get_ind(w, 'w')
            word_np[w_i, w_em] = 1
        # Process POS tag
        for t_i, t in enumerate(pos_):
            t_em = self.get_ind(t, 't')
            pos_np[t_i, t_em] = 1
        # Process label
        for l_i, l in enumerate(label_):
            l_em = self.get_ind(l, 'l')
            label_np[l_i, l_em] = 1

        return word_np, pos_np, label_np

class TrainData_large:
    """
    store the train data. Read all training data and prepocess them into some large numpy arrays.
    """
    def __init__(self, train_path):
        self.train_path = train_path
        self.n_lines = 0
        self.data = []

    def process_np(self, vocab):
        """
        process the list into np array
        """
        print "Processing the training data..."
        n_w, n_t, n_l, n_a = vocab.get_size()
        self.word = np.zeros((self.n_lines, 20, n_w))
        self.pos = np.zeros((self.n_lines, 20, n_t))
        self.label = np.zeros((self.n_lines, 12, n_l))
        self.action = np.zeros((self.n_lines, n_a))
        for i, s in enumerate(self.data):  ## Parsing each training examples
            fields = s.split(" ")
            assert len(fields) == 53
            word_ = fields[:20]
            pos_ = fields[20:40]
            label_ = fields[40:52]
            action_ = fields[-1]
            # Process word
            for w_i, w in enumerate(word_):
                w_em = vocab.get_ind(w, 'w')
                self.word[i, w_i, w_em] = 1
            # Process POS tag
            for t_i, t in enumerate(pos_):
                t_em = vocab.get_ind(t, 't')
                self.pos[i, t_i, t_em] = 1
            # Process label
            for l_i, l in enumerate(label_):
                l_em = vocab.get_ind(l, 'l')
                self.label[i, l_i, l_em] = 1
            # Process action
            a_em = vocab.get_ind(action_, 'a')
            self.action[i, a_em] = 1

    def get_data(self, vocab):
        """
        Get all the lines of the training data. Process them into np arrays.
        result in 4 np matrix:
            word:   (n_lines, 20, n_w)
            pos:    (n_lines, 20, n_t)
            label:  (n_lines, 12, n_l)
            action: (n_lines, n_a)
            n_w, n_t, n_l, n_a are from the Vocab Class
        """
        print "Reading lines from training data..."
        file = open(self.train_path, 'r')
        l = file.readline()
        while l:
            line = l.strip()
            self.n_lines += 1
            self.data.append(line)
            l = file.readline()
        print "Total training instance : %i" % self.n_lines
        self.process_np(vocab)

    def sample_batch(self, batchsize = 1000):
        """
        Sample a batch of lines from training data. Not good to use for training. Need to develop a iterator later.
        """
        idx = np.random.randint(self.n_lines, size=batchsize)
        return self.word[idx,:,:], self.pos[idx,:,:], self.label[idx,:,:], self.action[idx,:]

    def data_iter(self, batchsize = 1000):
        """
        Iterator to get all data in one epoch. Random shuffle is perform in each epoch.
        Copy first few items to make the whole length a multiple of batchsize
        """
        idx = np.random.permutation(self.n_lines)
        steps = self.n_lines / batchsize
        remain_len = (steps+1) * batchsize - self.n_lines
        remain = idx[:remain_len]
        idx = np.concatenate((idx, remain))
        for i in range(steps+1):
            ind_batch = idx[i*batchsize : (i+1)*batchsize]
            assert len(ind_batch) == batchsize
            yield [self.word[ind_batch,:,:], self.pos[ind_batch,:,:], self.label[ind_batch,:,:], self.action[ind_batch,:]]

class TrainData:
    """
    store the train data.
    """
    def __init__(self, train_path):
        self.train_path = train_path
        self.n_lines = 0
        self.data = []

    def process_np(self, vocab, idx):
        """
        process a batch of idx into np array from self.data
        """
        n_w, n_t, n_l, n_a = vocab.get_size()
        word = np.zeros((batchsize, 20, n_w))
        pos = np.zeros((batchsize, 20, n_t))
        label = np.zeros((batchsize, 12, n_l))
        action = np.zeros((batchsize, n_a))
        for i, s in enumerate(idx):  ## Parsing each training examples
            fields = self.data[s].split(" ")
            assert len(fields) == 53
            word_ = fields[:20]
            pos_ = fields[20:40]
            label_ = fields[40:52]
            action_ = fields[-1]
            # Process word
            for w_i, w in enumerate(word_):
                w_em = vocab.get_ind(w, 'w')
                word[i, w_i, w_em] = 1
            # Process POS tag
            for t_i, t in enumerate(pos_):
                t_em = vocab.get_ind(t, 't')
                pos[i, t_i, t_em] = 1
            # Process label
            for l_i, l in enumerate(label_):
                l_em = vocab.get_ind(l, 'l')
                label[i, l_i, l_em] = 1
            # Process action
            a_em = vocab.get_ind(action_, 'a')
            action[i, a_em] = 1

        return word,pos,label,action

    def get_data(self, vocab):
        """
        Get all the lines of the training data.
        Put them in a
        """
        print "Reading lines from training data..."
        file = open(self.train_path, 'r')
        l = file.readline()
        while l:
            line = l.strip()
            self.n_lines += 1
            self.data.append(line)
            l = file.readline()
        print "Total training instance : %i" % self.n_lines


    def data_iter(self, vocab, batchsize = 1000):
        """
        Iterator to get all data in one epoch. Random shuffle is perform in each epoch.
        Copy first few items to make the whole length a multiple of batchsize
        """
        idx = np.random.permutation(self.n_lines)
        steps = self.n_lines / batchsize
        remain_len = (steps+1) * batchsize - self.n_lines
        remain = idx[:remain_len]
        idx = np.concatenate((idx, remain))
        for i in range(steps+1):
            ind_batch = idx[i*batchsize : (i+1)*batchsize]
            assert len(ind_batch) == batchsize
            word, pos, label, action = self.process_np(vocab, ind_batch)
            yield word, pos, label, action


class Model():
    def __init__(self, vocab, mode = 'train'):
        self.vocab = vocab
        n_w, n_t, n_l, n_a = self.vocab.get_size()
        if mode == 'train':
            self.word = tf.placeholder(shape=[batchsize, 20, n_w], dtype=tf.float32)
            self.pos = tf.placeholder(shape=[batchsize, 20, n_t], dtype=tf.float32)
            self.label = tf.placeholder(shape=[batchsize, 12, n_l], dtype=tf.float32)
            self.action = tf.placeholder(shape=[batchsize, n_a], dtype=tf.float32)

            # Embedding Layers. No bias and activation.
            w = tf.layers.dense(inputs=self.word, units=word_em_dim, use_bias=False, activation=None)
            p = tf.layers.dense(inputs=self.pos, units=pos_em_dim, use_bias=False, activation=None)
            l = tf.layers.dense(inputs=self.label, units=label_em_dim, use_bias=False, activation=None)
            ## Embedding result. Dimension is batchsize * (20(d_w+d_t)+12d_l). By default is 1000*2304
            self.embed = tf.concat([tf.reshape(w, [batchsize, -1]), tf.reshape(p, [batchsize, -1]), tf.reshape(l, [batchsize, -1])], 1)

            ## Fully connect layer 1
            self.h1 = tf.layers.dense(inputs=self.embed, units=h1_dim, activation=tf.nn.relu)
            ## Fully connect layer 2
            self.h2 = tf.layers.dense(inputs=self.h1, units=h2_dim, activation=tf.nn.relu)
            ## Final layer
            self.result = tf.layers.dense(inputs=self.h2, units=n_a)
            ## softmax Loss
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.action, logits = self.result)

            ## Use Adam optimizer and Save logs.
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()

        elif mode == 'test':
            self.word = tf.placeholder(shape=[20, n_w], dtype=tf.float32)
            self.pos = tf.placeholder(shape=[20, n_t], dtype=tf.float32)
            self.label = tf.placeholder(shape=[12, n_l], dtype=tf.float32)

            # Embedding Layers. No bias and activation.
            w = tf.layers.dense(inputs=self.word, units=word_em_dim, use_bias=False, activation=None)
            p = tf.layers.dense(inputs=self.pos, units=pos_em_dim, use_bias=False, activation=None)
            l = tf.layers.dense(inputs=self.label, units=label_em_dim, use_bias=False, activation=None)
            ## Embedding result. Dimension is 1 * (20(d_w+d_t)+12d_l). By default is 1*2304
            self.embed = tf.concat([tf.reshape(w, [1, -1]), tf.reshape(p, [1, -1]), tf.reshape(l, [1, -1])], 1)

            ## Fully connect layer 1
            self.h1 = tf.layers.dense(inputs=self.embed, units=h1_dim, activation=tf.nn.relu)
            ## Fully connect layer 2
            self.h2 = tf.layers.dense(inputs=self.h1, units=h2_dim, activation=tf.nn.relu)
            ## Final layer
            self.result = tf.layers.dense(inputs=self.h2, units=n_a)
            ## do softmax to get scores
            self.score = tf.nn.softmax(self.result)

        self.saver = tf.train.Saver()

    def train(self, train_data, path, sess):
        sess.run(tf.global_variables_initializer())
        model_path = os.path.join(path,'check_points')
        log_path = os.path.join(path,'logs')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(model_path):
            os.makedirs(log_path)

        writer = tf.summary.FileWriter(log_path, sess.graph)

        for i in range(epoch):
            print "Epoch %i:" % i
            count = 1
            loss_acc = 0
            for batch in train_data.data_iter(self.vocab, batchsize = batchsize):
                _, loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict={self.word: batch[0], self.pos:batch[1], self.label: batch[2], self.action: batch[3]})

                loss_acc += loss
                if (count) % show_iter == 0 or count == 1:
                    print(' (Iter : %d) Loss: %.4f' % (count, loss_acc/count ))
                    loss_acc = 0
                    writer.add_summary(summary, i*143+count)

                count += 1

            self.saver.save(sess, os.path.join(model_path, 'model.ckpt'), global_step=i)  ## Saving for Each epoch
            print('model is saved')

    def load(self, model_path, sess):
        print "Loading model from %s..." % model_path
        self.saver.restore(sess, model_path)

    def predict(self, sess, input):
        """
        return a numpy array of score
        """
        n_w, n_t, n_l, _ = self.vocab.get_size()
        word, pos, label = self.vocab.parse_input(input, n_w, n_t, n_l)
        pred_score = sess.run([self.score], feed_dict={self.word: word, self.pos:pos, self.label: label})
        return pred_score[0]



if __name__ == '__main__':
    vocab = Vocab(vocabs_dir)
    vocab.get_dic()

    train_data = TrainData(train_path)
    train_data.get_data(vocab)

    model = Model(vocab)
    sess = tf.Session()
    model.train(train_data, model_dir, sess)
