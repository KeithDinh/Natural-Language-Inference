import os
import sys
import time
import argparse
from datetime import timedelta
from collections import Counter
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import tensorflow.keras as keras

UNKNOWN = '<<UNK>>'
PADDING = '<<PAD>>'
CATEGORIE_ID = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

tf.disable_v2_behavior()

def load_dataset(file_path):
    """
    Loading dataset and dropping unused features
    :param file_path: dataset needed to load
    :return: a clean dataset which is ready for training or further processing
    """
    dataset = pd.read_csv(file_path)
    dataset = dataset.drop(
        ['sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'label1', 'label2',
         'label3', 'label4', 'label5'], axis=1)
    print('Dataset shape:', dataset.shape)
    print('Loading dataset completed.')
    return dataset


def transform_embedding_to_txt(file_path, num_words, embed_size):
    """
    Converting the pre-train file into a text file with a .we extension.
    This file contains a 2 dimensional numpy array (embeddings and dictionary)
    """
    vocab = {}
    wid = 0
    wrong = 0
    embeddings = np.zeros((num_words, embed_size), dtype=np.float32)
    file = open(file_path, 'r', encoding='utf-8', errors='ignore')
    for line in file:
        items = line.strip().split()
        if len(items) != embed_size + 1:
            wrong += 1
            print(line)
            continue

        if items[0] in vocab:
            wrong += 1
            print(line)
            continue

        vocab[items[0]] = wid
        embeddings[wid] = [float(it) for it in items[1:]]
        wid += 1

    # dump
    save_path = file_path.rsplit('.', 1)[0] + '.we'
    embeddings = embeddings[0:wid, ]
    with open(save_path, 'wb') as fout:
        pickle.dump([embeddings, vocab], fout)

    print(len(vocab), embeddings.shape, 'wrong words: ', wrong, 'total words: ', num_words)
    print("Save in: ", save_path)


def load_embeddings(embdding_path, vocab):
    """
    Loading pre-trained model
    """
    file = open(embdding_path, 'rb')
    _embeddings, _vocab = pickle.load(file)
    embedding_size = _embeddings.shape[1]

    embeddings = create_embeddings(vocab, embedding_size)
    for word, id in vocab.items():
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]

    return embeddings.astype(np.float32)


def create_embeddings(vocab, embedding_size):
    """
    Initialize word embeddings with an initializable size
    """
    rng = np.random.RandomState()
    embeddings = rng.normal(loc=0.0, scale=1.0, size=(len(vocab), embedding_size))
    return embeddings.astype(np.float32)


def embedding_normalization(embeddings):
    """
    Perform normalization on the embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


def open_file(file_path, mode='r'):
    """
    return a opened file with write mode, encoding uft-8
    """
    return open(file_path, mode, encoding='utf-8', errors='ignore')


def build_vocab(file_path, vocab_path, nfreq=0, lowercase=True):
    """
    Building a vocabulary data using training file
    word ||| frequency.
    If the word's frequency < nfreq. Eliminate the word.
    """
    counter = Counter()
    # read train data file
    file = open_file(file_path)
    for line in file:
        try:
            if lowercase:
                line = line.lower()
            words = line.strip().split()
            for word in list(words):
                counter[word] += 1
        except:
            pass

    count_pairs = [item for item in counter.items() if item[1] >= nfreq]
    count_pairs = sorted(count_pairs, key=lambda k: k[1], reverse=True)
    word_freqs = [' ||| '.join([w, str(f)]) for w, f in count_pairs]
    open_file(vocab_path, mode='w').write('\n'.join(word_freqs) + '\n')
    print('Vocabulary is stored in: {0}'.format(vocab_path))


def load_vocab(vocab_path, cut_off=0, adding=True):
    """
    Loading vocabulary data
    Remove word's frequency < cut_off
    If adding=True, we add<<UNK>> and <<PAD>>
    """
    vocab = {}
    idx = 0
    if adding:
        vocab[PADDING] = 0
        vocab[UNKNOWN] = 1
        idx = 2
    file =  open(vocab_path, encoding='utf-8')
    for line in file:
        items = line.split('|||')
        if len(items) != 2:
            print('Wrong format: ', line)
            continue
        word, freq = line.split('|||')
        word, freq = word.strip(), int(freq.strip())
        if freq >= cut_off:
            vocab[word] = idx
            idx += 1
    return vocab


def print_log_file(*args, **keyargs):
    """
    Print on both terminal and log file
    """
    print(*args)
    if len(keyargs) > 0:
        print(*args, **keyargs)
    return None


def print_hyper_parameters(args, log_f):
    """
    Print all used parameters on both terminal and log file
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log_file("------------- HYPER PARAMETERS -------------", file=log_f)

    for a in argsList:
        print_log_file("%s: %s" % (a[0], str(a[1])), file=log_f)
    print("-----------------------------------------", file=log_f)
    return None


def count_parameters():
    """
    Return the number of trainable parameters
    """
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim.value
        total_params += variable_params
    return total_params


def get_time(start_time):
    """
    Calculating time spending
    """
    end_time = time.time()
    time_spend = end_time - start_time
    return timedelta(seconds=int(round(time_spend)))


def transform_data(data_file, vocab, cat_id=CATEGORIE_ID, max_len1=50, max_len2=50, lowercase=True):
    """
    Processing data by using data and vocab file. Then, return padding, mask of each sentence and label

    """
    sentence1_id, sentence2_id, label_id = [], [], []
    file = open_file(data_file)
    for line in file:
        try:
            label, sentence1, sentence2 = [x.strip() for x in line.strip().split('|||')]
            if lowercase:
                sentence1 = sentence1.lower()
                sentence2 = sentence2.lower()
            sentence1 = [x.strip() for x in sentence1.split()]
            sentence2 = [x.strip() for x in sentence2.split()]
            if label in cat_id:
                sentence1_id.append([vocab[x] if x in vocab else vocab[UNKNOWN] for x in sentence1])
                sentence2_id.append([vocab[x] if x in vocab else vocab[UNKNOWN] for x in sentence2])
                label_id.append(cat_id[label])
        except:
            ValueError('Value error!')

    # Convert the sequence into a fixed length padding
    sentence1_padding = keras.preprocessing.sequence.pad_sequences(sentence1_id, max_len1, padding='post')
    sentence2_padding = keras.preprocessing.sequence.pad_sequences(sentence2_id, max_len2, padding='post')
    # Compute masks of each sentence
    sentence1_mask = (sentence1_padding > 0).astype(np.int32)
    sentence2_mask = (sentence2_padding > 0).astype(np.int32)
    # Padding the label before returning
    label_padding = np.asarray(label_id, np.int32)
    return sentence1_padding, sentence1_mask, sentence2_padding, sentence2_mask, label_padding


def next_batch(s1, s1_mask, s2, s2_mask, y_pad, batch_size=64, shuffle=True):
    """Generating bath of data using output from transform_data function"""
    data_len = len(s1)
    num_batch = int((data_len - 1) / batch_size) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
        s1 = s1[indices]
        s1_mask = s1_mask[indices]
        s2 = s2[indices]
        s2_mask = s2_mask[indices]
        y_pad = y_pad[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield (s1[start_id:end_id], s1_mask[start_id:end_id],
               s2[start_id:end_id], s2_mask[start_id:end_id],
               y_pad[start_id:end_id])


class SNLIModel(object):
    '''
    A class store all the hyper-parameters and set function for model
    '''
    def __init__(self, n_classes, vocab_size, embedding_size, mlen1, mlen2, vocab,
                 attend_layer_sizes, compare_layer_sizes, aggregate_layer_sizes, proj_emb_size,
                 optimizer_algorithm='adagrad', train_em=True, proj_emb=False):
        # hyper-parameters
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.attend_layer_sizes = attend_layer_sizes
        self.compare_layer_sizes = compare_layer_sizes
        self.aggregate_layer_sizes = aggregate_layer_sizes
        self.proj_emb_size = proj_emb_size
        self.optimizer_algorithm = optimizer_algorithm
        self.train_em = train_em
        self.proj_emb = proj_emb
        self.mlen1 = mlen1
        self.mlen2 = mlen2

        # Require for running test later
        _, _, _ = tf.constant(mlen1, name='mlen1'), tf.constant(mlen2, name='mlen2'), \
                  tf.constant(vocab, name='vocab')

        # Initialize data from tensorflow placeholder
        # Data and masks of sentence 1
        self.s1 = tf.placeholder(tf.int32, (None, self.mlen1), name='sen1')
        self.s1_m = tf.placeholder(tf.float32, (None, self.mlen1), name='sen1_mask')
        # Data and masks of sentence 2
        self.s2 = tf.placeholder(tf.int32, (None, self.mlen2), name='sen2')
        self.s2_m = tf.placeholder(tf.float32, (None, self.mlen2), name='sen2_mask')
        # gold labels
        self.y = tf.placeholder(tf.int32, (None), name='n_classes')

        # Obtain learning rate, L2 loss, dropout and clip value
        self.lr = tf.placeholder(tf.float32, (), name='learning_rate')
        self.l2 = tf.placeholder(tf.float32, (), name='l2_contant')
        self.dropout_keep = tf.placeholder(tf.float32, (), name='dropout_keep')
        self.clip_value = tf.placeholder(tf.float32, (), name='clip_value')

        # initialize the embedding from a tensorflow placeholder
        self.embeddings_ph = tf.placeholder(tf.float32,
                                            (self.vocab_size, self.embedding_size))
        self.embeddings = tf.Variable(self.embeddings_ph, trainable=self.train_em,
                                      validate_shape=True, name='embeddings')
        # Build graph
        self.build_graph()

    def build_graph(self):

        def _transform_embeddings(emb, num_unit, reuse_weights=False):
            """
            Transform the embedding into different dimension.
            Return embeddings with new shape (batch, mlen, num_unit)
            """
            with tf.variable_scope('proj_emb', reuse=reuse_weights) as self.proj_scope:
                initializer = tf.random_normal_initializer(0.0, 0.1)
                projected = tf.layers.dense(emb, num_unit, kernel_initializer=initializer)
            return projected

        def _feedforwad(inputs, scope, num_units,
                               reuse_weights=False,
                               initializer=None):
            """
            Feed two feed-forward layers with num_units on the inputs.
            """
            scope = scope or 'feedforward'
            with tf.variable_scope(scope, reuse=reuse_weights):
                if initializer is None:
                    initializer = tf.random_normal_initializer(0.0, 0.1)

                with tf.variable_scope('layer1'):
                    inputs = tf.nn.dropout(inputs, self.dropout_keep)
                    relus1 = tf.layers.dense(inputs, num_units[0], tf.nn.relu, kernel_initializer=initializer)
                with tf.variable_scope('layer2'):
                    inputs = tf.nn.dropout(relus1, self.dropout_keep)
                    relus2 = tf.layers.dense(inputs, num_units[1], tf.nn.relu, kernel_initializer=initializer)
            return relus2

        def _attention(sent1, sent2):
            """
            Compute attention of sentences, positional encoding
             """
            with tf.variable_scope('attend_scope') as self.attend_scope:
                num_units = self.attend_layer_sizes

                repr1 = _feedforwad(sent1, self.attend_scope, num_units)
                repr2 = _feedforwad(sent2, self.attend_scope, num_units, reuse_weights=True)

                m1_m2 = tf.multiply(tf.expand_dims(self.s1_m, 2), tf.expand_dims(self.s2_m, 1))

                repr2 = tf.transpose(repr2, [0, 2, 1])
                origin_attention = tf.matmul(repr1, repr2)
                origin_attention = tf.multiply(origin_attention, m1_m2)

                attention1 = tf.exp(origin_attention - tf.reduce_max(origin_attention, axis=2, keep_dims=True))
                attention2 = tf.exp(origin_attention - tf.reduce_max(origin_attention, axis=1, keep_dims=True))

                attention1 = tf.multiply(attention1, tf.expand_dims(self.s2_m, 1))
                attention2 = tf.multiply(attention2, tf.expand_dims(self.s1_m, 2))

                attention1 = tf.divide(attention1, tf.reduce_sum(attention1, axis=2, keep_dims=True))
                attention2 = tf.divide(attention2, tf.reduce_sum(attention2, axis=1, keep_dims=True))

                attention1 = tf.multiply(attention1, m1_m2)
                attention2 = tf.multiply(attention2, m1_m2)

                alpha = tf.matmul(attention1, sent2, name='alpha')
                beta = tf.matmul(tf.transpose(attention2, [0, 2, 1]), sent1, name='beta')

            return alpha, beta

        def _compare(sen, soft_align, reuse_weights=False):
            '''
            Compare one sentence to its soft alignment
            with the other by using a feed-forward.
            '''
            with tf.variable_scope('compare_score', reuse=reuse_weights) as self.comapre_score:
                inputs = [sen, soft_align]
                inputs = tf.concat(inputs, axis=2)
                # two-layer NN
                num_units = self.compare_layer_sizes
                output = _feedforwad(inputs, self.comapre_score,
                                            num_units, reuse_weights)

            return output

        def _aggregate(v1, v2):
            """
            Aggregate the representations induced from both sentences and their
            representations
            Note that: No masks are used.
            :param v1: tensor with shape (batch, mlen1, num_unit_input1)
            :param v2: tensor with shape (batch, mlen2, num_unit_input2)
            :return: logits over classes, shape (batch, n_classes)
            """
            # calculate sum
            v1_sum = tf.reduce_sum(v1, 1)
            v2_sum = tf.reduce_sum(v2, 1)

            inputs = tf.concat(axis=1, values=[v1_sum, v2_sum])
            with tf.variable_scope('aggregate_scope') as self.aggregate_scope:
                num_units = self.aggregate_layer_sizes
                logits = _feedforwad(inputs, self.aggregate_scope, num_units)
                # the last layer
                logits = tf.layers.dense(logits, self.n_classes, name='last_layer')

            return logits

        def _create_training_op(optimizer_algorithm):
            """
            Create the operation used for training
            """
            with tf.name_scope('training'):
                if optimizer_algorithm == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(self.lr)
                elif optimizer_algorithm == 'adam':
                    optimizer = tf.train.AdamOptimizer(self.lr)
                elif optimizer_algorithm == 'adadelta':
                    optimizer = tf.train.AdadeltaOptimizer(self.lr)
                else:
                    ValueError('Unkown optimizer: {0}'.format(optimizer_algorithm))

            # clip gradients
            gradients, v = zip(*optimizer.compute_gradients(self.loss))

            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            train_op = optimizer.apply_gradients(zip(gradients, v))
            return train_op

        # build graph
        with tf.device('/cpu:0'):
            emb1 = tf.nn.embedding_lookup(self.embeddings, self.s1)
            emb2 = tf.nn.embedding_lookup(self.embeddings, self.s2)

        # the architecture has 3 main steps: soft align, compare and aggregate
        with tf.name_scope('align_compare_aggregate'):
            if self.proj_emb:
                repr1 = _transform_embeddings(emb1, self.proj_emb_size)
                repr2 = _transform_embeddings(emb2, self.proj_emb_size, reuse_weights=True)
            else:
                repr1, repr2 = emb1, emb2

            alpha, beta = _attention(repr1, repr2)

            repr1 = tf.multiply(repr1, tf.expand_dims(self.s1_m, -1))
            repr2 = tf.multiply(repr2, tf.expand_dims(self.s2_m, -1))
            v1, v2 = _compare(repr1, alpha), _compare(repr2, beta, reuse_weights=True)

            self.logits = _aggregate(v1, v2)

        # Training
        with tf.name_scope('optimize'):
            # Classification loss
            cross_entropy = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            labeled_loss = tf.reduce_mean(cross_entropy)
            # L2 loss
            weights = [v for v in tf.trainable_variables() if 'kernel' in v.name]
            l2_partial_sum = sum([tf.nn.l2_loss(weight) for weight in weights])
            l2_loss = tf.multiply(self.l2, l2_partial_sum)
            # total loss = classification loss + L2 loss
            self.loss = tf.add(labeled_loss, l2_loss)
            self.train_op = _create_training_op(self.optimizer_algorithm)

        # Testing / predicting
        with tf.name_scope('predict'):
            self.y_pred = tf.cast(tf.argmax(tf.nn.softmax(self.logits), axis=1), tf.int32, name='y')
            num_correct_predict = tf.equal(self.y, self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(num_correct_predict, tf.float32))
        # End build graph funciton


def feeding_data(s1_batch, s1_batch_mask, s2_batch, s2_batch_mask, y_batch,
              learning_rate, dropout_keep, l2, clip_value):
    feed_dict = {
        model.s1: s1_batch,
        model.s1_m: s1_batch_mask,
        model.s2: s2_batch,
        model.s2_m: s2_batch_mask,
        model.y: y_batch,
        model.lr: learning_rate,
        model.dropout_keep: dropout_keep,
        model.l2: l2,
        model.clip_value: clip_value
    }
    return feed_dict


def evaluate_performance(sess, s1, s1_mask, s2, s2_mask, y):
    """evaluate performance the performance on a data set
    return loss and accuracy
    """
    data_len = len(s1)
    batch_eval = next_batch(s1, s1_mask, s2, s2_mask, y, config_dict['batch_size'], shuffle=False)
    total_loss = 0.0
    total_acc = 0.0
    for batch in batch_eval:
        s1_batch = batch[0]
        batch_len = len(s1_batch)
        feed_dict = feeding_data(*batch, config_dict['learning_rate'], 1.0,
                              config_dict['L2'] , config_dict['clip_value'])
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


# A dictionary store the configuration of hyper-paramters,
# file paths, embedding layer setup, etc
config_dict = {
    # hyper-parameters
    'num_epoch': 1, # Number of epochs
    'batch_size': 500, # Batch size
    'dropout': 0.8, # Dropout keep prob
    'clip_value': 5.0, # Normal clip training gradients
    'learning_rate': 0.1, # Learning rate
    'L2': 0.001,
    'max_length_sen1': 50,
    'max_length_sen2': 50,
    'optimizer': 'adagrad', # 'adagrad', 'adadelta', 'adam'
    'num_class': 3,

    # file path
    'log': 'log/',
    'vocab_txt': 'vocab.txt',
    'train_csv': 'training_clean.csv',
    'dev_csv': 'dev_clean.csv',
    'test_csv': 'testing_clean.csv',
    'train_txt': 'train.txt',
    'dev_txt': 'dev.txt',
    'test_txt': 'test.txt',
    'pretrained_glove_txt': 'glove.840B.300d.txt', # Pre-trained GloVe word embedding file
    'glove_we': 'glove.840B.300d.we',

    # embedding layer setup
    'cut_off': 0, #If the freq of a word < cut_off, we remove it
    'embedding_size': 50,
    'norm_word_embedding' : 1,
    'train_em': 0, # fine-tuning word embeddings

    # Other layer configuration
    'attend_layer_size': [200,200],
    'compare_layer_size': [200,200],
    'aggregate_layer_size': [200,200],

    # projecting word embeddings or not
    'project_word_embedding': 1,
    'project_word_embedding_size': 200,

    # save and report settings
    'save_model_file': 'model/SNLI',
    'num_of_batch_btw_performance_reports': 500,
    'save_per_batch': 500, # num of batches btw savaing to tesorboard scalar
    'req_improvement': 100000,
    'tf_board': 'data/',

}

# Start preparing for training
if __name__ == '__main__':
    # Use a parser for writing hyper-parameter into log file
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()
    args.num_epoch = config_dict['num_epoch']
    args.batch_size = config_dict['batch_size']
    args.drop_out = config_dict['dropout']
    args.clip_value = config_dict['clip_value']
    args.learning_rate = config_dict['learning_rate']
    args.L2 = config_dict['L2']
    args.max_length_sen1 = config_dict['max_length_sen1']
    args.max_length_sen2 = config_dict['max_length_sen2']
    args.optimizer = config_dict['optimizer']
    args.num_class = config_dict['num_class']

    # Execute this function will transform our csv data file into a txt file format
    # transfer_data('dev_clean.csv', 'dev.txt')
    # transfer_data('testing_clean.csv', 'test.txt')
    # transfer_data('training_clean.csv', 'train.txt')

    # Function transform_embedding_to_txt and build_vocab only
    # need to be executed in the first time. Later on, just
    # skip both functions

    # Execute this function to transform the embedding to
    # a convinience text file
    # transform_embedding_to_txt('glove.840B.300d.txt', 2196017, 300)

    #Build vocab file using pre-trained GloVe and training data
    build_vocab('train.txt', 'vocab.txt')

    # load vocabulary and embeddings
    vocab = load_vocab(config_dict['vocab_txt'], cut_off=config_dict['cut_off'], adding=True)
    args.vocab_size = len(vocab)
    if config_dict['glove_we']:
        # load pre-trained model
        embeddings = load_embeddings(config_dict['glove_we'], vocab)
        config_dict['embedding_size'] = embeddings.shape[1]
    else:
        # Initialize word embeddings randomly
        embeddings = create_embeddings(vocab, config_dict['embedding_size'])
    if config_dict['norm_word_embedding']:
        embeddings = embedding_normalization(embeddings)

    config_dict['train_em'] = config_dict['train_em'] != 0
    config_dict['project_word_embedding'] = config_dict['project_word_embedding'] != 0
    # Print hyper-parameters on terminal and write log file
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.log = "log/SNLI/log.{0}".format(dt)
    log = open(args.log, 'w')
    print_log_file('CMD: python3 {0}'.format(' '.join(sys.argv)), file=log)
    print_log_file('Training with following options:', file=log)
    print_hyper_parameters(args, log)

    # Building model
    model = SNLIModel(config_dict['num_class'], args.vocab_size,
                                config_dict['embedding_size'], config_dict['max_length_sen1'],
                                config_dict['max_length_sen2'], config_dict['vocab_txt'],
                                config_dict['attend_layer_size'],
                                config_dict['compare_layer_size'],
                                config_dict['aggregate_layer_size'],
                                config_dict['project_word_embedding_size'],
                                config_dict['optimizer'],
                                train_em=config_dict['train_em'],
                                proj_emb=config_dict['project_word_embedding'])

    print_log_file("Loading training and validation data...", file=log)
    start_time = time.time()
    s1_train, s1_train_mask, s2_train, s2_train_mask, y_train = transform_data(
        config_dict['train_txt'], vocab,
        max_len1=config_dict['max_length_sen1'],
        max_len2=config_dict['max_length_sen2'], lowercase=True)
    s1_val, s1_val_mask, s2_val, s2_val_mask, y_val = transform_data(
        config_dict['dev_txt'], vocab,
        max_len1=config_dict['max_length_sen1'],
        max_len2=config_dict['max_length_sen2'], lowercase=True)

    data_len = len(s1_train)
    time_spend = get_time(start_time)
    print_log_file("Time usage:", time_spend, file=log)

    # Tensorboard
    print_log_file("Configuring TensorBoard and Saver...", file=log)
    if not os.path.exists(config_dict['tf_board']):
        os.makedirs(config_dict['tf_board'])
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config_dict['tf_board'])
    # Saving model
    saver = tf.train.Saver()
    save_dir, _ = os.path.split(config_dict['save_model_file'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), {model.embeddings_ph: embeddings})
    writer.add_graph(sess.graph)
    # Count trainable parameters
    total_params = count_parameters()
    print_log_file('Total parameters: {0}'.format(total_params), file=log)

    # Start training and evaluate
    print_log_file('Training and evaluating...', file=log)
    start_time = time.time()
    total_batch = 0  # total batches
    best_acc_val = 0.0  # the best accuracy on validation set
    last_improved = 0  # the last batch with improved accuracy
    flag = False  # stop training
    for epoch in range(config_dict['num_epoch']):
        print_log_file('Epoch:', epoch + 1, file=log)
        batch_train = next_batch(s1_train, s1_train_mask,
                                          s2_train, s2_train_mask,
                                          y_train,
                                          config_dict['batch_size'],
                                          shuffle=True)
        total_loss, total_acc = 0.0, 0.0
        for batch in batch_train:
            batch_len = len(batch[0])
            # s1_batch, s1_batch_mask, s2_batch, s2_batch_mask, y_batch = batch
            feed_dict = feeding_data(*batch, config_dict['learning_rate'],
                                  config_dict['dropout'],
                                  config_dict['L2'],
                                  config_dict['clip_value'])
            # Optimize, obtain the loss and accuracy on the current training batch
            _, batch_loss, batch_acc = sess.run([model.train_op, model.loss, model.acc], feed_dict=feed_dict)
            total_loss += batch_loss * batch_len
            total_acc += batch_acc * batch_len

            if total_batch % config_dict['save_per_batch'] == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config_dict['num_of_batch_btw_performance_reports'] == 0:
                feed_dict[model.dropout_keep] = 1.0
                # Validation's loss and accuracy of the whole training time
                loss_val, acc_val = evaluate_performance(sess, s1_val, s1_val_mask, s2_val, s2_val_mask, y_val)
                if acc_val > best_acc_val:
                    # Store the best model indo model folder
                    # We can use model for testing later
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=config_dict['save_model_file'], )
                    improved_str = '*'
                else:
                    improved_str = ''
                # Report training process
                time_spend = get_time(start_time)
                msg = 'Epoch: {0:>2}, Batch: {1:>7}, Train Batch Loss: {2:>6.2}, Train Batch Acc: {3:>7.2%},' \
                      + ' Val Loss: {4:>6.2}, Val Acc: {5:>7.2%}, Time: {6} {7}'
                print_log_file(msg.format(epoch + 1, total_batch, batch_loss, batch_acc, loss_val,
                                   acc_val, time_spend, improved_str), file=log)

            total_batch += 1
            if total_batch - last_improved > config_dict['req_improvement']:
                # Early stopping if there is no improvement
                print_log_file("No optimization for a long time, auto-stopping...", file=log)
                flag = True
                break
        if flag:
            break
        # Reporting time usage, training loss, and training accuracy at the
        # end of each epoch iteration
        time_spend = get_time(start_time)
        total_loss, total_acc = total_loss / data_len, total_acc / data_len
        msg = '*** Epoch: {0:>2}, Train Loss: {1:>6.2}, Train Acc: {2:7.2%}, Time: {3}'
        print_log_file(msg.format(epoch + 1, total_loss, total_acc, time_spend), file=log)

    sess.close()
    log.close()
