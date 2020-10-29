import sys
import time
import keras
import argparse
import warnings
import tempfile
import numpy as np
from os import path
import tensorflow as tf
import keras.backend as K
from sklearn import metrics
from datetime import datetime
from datetime import timedelta
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import recurrent, Dense, Input, Dropout, TimeDistributed, concatenate


warnings.filterwarnings('ignore')
label_category = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

class Configuration:
    # tuning parameters
    RNN = recurrent.LSTM
    num_layer = 1
    is_GLoVe_enable = True
    is_train_embedding = False
    embedding_hidden_size = 300
    sent_hidden_size = 300
    batch_size = 512
    patience = 4 # 8
    num_epoch = 20
    max_length = 42
    dropout = 0.2
    L2 = 4e-6
    activation_function = 'relu'
    optimizer = 'rmsprop' # rmsprop, adagrad, adadelta, adam
    GloVe_weights = 'GloVe_weights'
    labels = label_category
    num_line = None
    train = 'train.txt'
    dev = 'dev.txt'
    test = 'test.txt'


def retrieve_labels_sentences(fileName, limitNumOfLine=None):
    """
    Loading txt file and processing data.
    :param fileName: text file
    :param limitNumOfLine:
    :return: list of sen1, sen2, binning_label
    """
    print('Loading', fileName, ' file .......')
    # print_log_file('Loading {0} file .......'.format(fileName), file=log)
    file = open(fileName, 'r', encoding='utf-8', errors='ignore')
    
    sen1, sen2, labels = [], [], []
    # print_log_file('Retrieving data .......', file=log)
    print('Retrieving data .......')
    for line in file:
        label, sentence1, sentence2 = [x.strip() for x in line.strip().split('|||')]
        
        if sentence1[-1] is '.':
            sentence1 = sentence1[:-1] + ' .'
        if sentence2[-1] is '.':
            sentence2 = sentence2[:-1] + ' .'

        sen1.append(sentence1)
        sen2.append(sentence2)
        labels.append(label)

        global label_category
    file.close()

    list_labels = np.array([label_category[each_row] for each_row in labels])
    binning_label = np_utils.to_categorical(list_labels, len(label_category))
    # print_log_file('Retrieving {0} data completed .......\n'.format(fileName), file=log)
    print('Retrieving', fileName, ' data completed .......\n')
    return sen1, sen2, binning_label


# texts_to_sequences convert sentence to sequence of numbers
# pad_sequences fill the missing position with 0
def text_to_seq(data):
    sen1_to_seq = pad_sequences(tokenizer.texts_to_sequences(data[0]), maxlen=config.max_length)
    sen2_to_seq = pad_sequences(tokenizer.texts_to_sequences(data[1]), maxlen=config.max_length)
    return (sen1_to_seq,sen2_to_seq,data[2])


def print_log_file(*args, **keyargs):
    """
    Print on both terminal and log file
    """
    print(*args)
    if len(keyargs) > 0:
        print(*args, **keyargs)
    return None


def print_hyper_parameters(args, log_file):
    """
    Print all used parameters on both terminal and log file
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log_file("------------- HYPER PARAMETERS -------------", file=log_file)

    for a in argsList:
        print_log_file("%s: %s" % (a[0], str(a[1])), file=log_file)
    print("-----------------------------------------", file=log_file)
    return None

if __name__ == '__main__':
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('Loading configuration .......')
    config = Configuration()
    # print_log_file('Loading configuration .......', file=log)

    # Create a parser for storing hyper-parameter into log file
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()
    args.pretrained_GloVe_model = config.is_GLoVe_enable
    args.train_embedding = config.is_train_embedding
    args.embedding_hidden_layer_size = config.embedding_hidden_size
    args.sent_hidden_layer_size = config.sent_hidden_size
    args.batch_size = config.batch_size
    args.patience = config.patience
    args.num_of_epochs = config.num_epoch
    args.max_length = config.max_length
    args.dropout = config.dropout
    args.L2 = config.L2
    args.activation_funciton = config.activation_function
    args.optimizer = config.optimizer
    args.GloVe_weights = config.GloVe_weights
    args.labels = config.labels
    args.num_of_line = config.num_line
    args.rnn = config.RNN

    # Print hyper-parameters on CMD and write log file
    args.log = "log/SNLI_LSTM_GRU/hyper-parameters.{0}".format(dt)
    log = open(args.log, 'w')
    print_log_file('CMD: python3 {0} \n'.format(' '.join(sys.argv)), file=log)


    # Loading train, dev, and test data
    train = retrieve_labels_sentences(config.train, limitNumOfLine=config.num_line)
    dev = retrieve_labels_sentences(config.dev, limitNumOfLine=config.num_line)
    test = retrieve_labels_sentences(config.test, limitNumOfLine=config.num_line)

    # Combine two lists to get the amount of each distinct word in all sentences
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(train[0] + train[1])
    # amount_distinct_words = len(tokenizer.word_counts)
    vocab = len(tokenizer.word_counts) + 1
    args.vocab_size = vocab
    # Convert train, dev, test to sequences
    train_sequence = text_to_seq(train)
    test_sequence = text_to_seq(test)
    dev_sequence = text_to_seq(dev)

    print_log_file('Training with following options:', file=log)
    print_hyper_parameters(args, log)

    if config.is_GLoVe_enable:
        if path.exists(config.GloVe_weights + '.npy'):
            # print_log_file('Found {0}.npy'.format(config.GloVe_weights), file=log)
            print('Found', config.GloVe_weights+'.npy')
            # print_log_file('Loading weights from the file .......', file=log)
            print('Loading weights from the file .......')
            embedding_matrix = np.load(config.GloVe_weights + '.npy')
        else:
            # print_log_file('File {0}.npy not found.'.format(config.GloVe_weights), file=log)
            print('File', config.GloVe_weights, '.npy not found.')
            # print_log_file('Loading glove.840B.300d.txt .......', file=log)
            print('Loading glove.840B.300d.txt .......')
            embeddings_index = {}
            file = open('glove.840B.300d.txt','r', encoding='utf-8', errors='ignore')
            # print_log_file('Computing GloVe from file .......', file=log)
            print('Computing GloVe from file .......')
            for line in file:
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            file.close()

            # prepare embedding matrix
            embedding_matrix = np.zeros((vocab, config.embedding_hidden_size))
            missing_words = list()
            for word, i in tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                else:
                    missing_words.append(word)

            # print_log_file('Total number missing word from GloVe: {0}'.format(len(missing_words)), file=log)
            print('Total number missing word from GloVe:', len(missing_words))
            # print(missing_words, '\n')
            # print_log_file('Saving GloVe_weights.npy completed .......', file=log)
            print('Saving GloVe_weights.npy completed .......')
            np.save(config.GloVe_weights, embedding_matrix)

        # print_log_file('Total number of null word embeddings: {0} \n'.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)), file=log)
        print('Total number of null word embeddings:', np.sum(np.sum(embedding_matrix, axis=1) == 0), '\n')

        # print_log_file('Constructing embedding layer .......', file=log)
        print('Constructing embedding layer .......')
        embedding_layer = Embedding(vocab, config.embedding_hidden_size, weights=[embedding_matrix], input_length=config.max_length, trainable=config.is_train_embedding)
    else:
        embedding_layer = Embedding(vocab, config.embedding_hidden_size, input_length=config.max_length)


    rnn_args = dict(output_dim=config.sent_hidden_size, dropout_W=config.dropout, dropout_U=config.dropout)
    Add_Embeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(config.sent_hidden_size,))

    translate = TimeDistributed(Dense(config.sent_hidden_size, activation=config.activation_function))

    # print_log_file('Constructing premise & hypothesis layers .......', file=log)
    print('Constructing premise & hypothesis layers .......')
    premise_layer = Input(shape=(config.max_length,), dtype='int32')
    hypothesis_layer = Input(shape=(config.max_length,), dtype='int32')

    premise = embedding_layer(premise_layer)
    hypothesis = embedding_layer(hypothesis_layer)

    premise = translate(premise)
    hypothesis = translate(hypothesis)

    rnn = Add_Embeddings if not config.RNN else config.RNN(return_sequences=False, **rnn_args)
    premise = rnn(premise)
    hypothesis = rnn(hypothesis)
    premise = BatchNormalization()(premise)
    hypo = BatchNormalization()(hypothesis)

    joint = concatenate([premise, hypothesis])
    joint = Dropout(config.dropout)(joint)

    for i in range(3):
        joint = Dense(2 * config.sent_hidden_size, activation=config.activation_function, W_regularizer=l2(config.L2) if config.L2 else None)(joint)
        joint = Dropout(config.dropout)(joint)
        joint = BatchNormalization()(joint)

    # print_log_file('Constructing output layer .......', file=log)
    print('Constructing output layer .......')
    output_layer = Dense(len(config.labels), activation='softmax')(joint)
    # print_log_file('All layers are completely constructed .......\n', file=log)
    print('All layers are completely constructed .......\n')

    # print_log_file('Building model .......\n', file=log)
    print('Building model .......\n')
    SNLIModel = Model(input=[premise_layer, hypothesis_layer], output=output_layer)
    SNLIModel.compile(optimizer=config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    SNLIModel.summary()


    # print_log_file('Start training process .......', file=log)
    print('Start training process .......')
    start_time = time.time()
    _, tmpfn = tempfile.mkstemp()
    callbacks = [EarlyStopping(patience=config.patience), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
    history = SNLIModel.fit([train_sequence[0], train_sequence[1]], train_sequence[2], batch_size=config.batch_size, nb_epoch=config.num_epoch, validation_data=([dev_sequence[0], dev_sequence[1]], dev_sequence[2]), callbacks=callbacks)

    SNLIModel.load_weights(tmpfn)

    # print_log_file('\n\nRunning test  .......\n', file=log)
    print('\nRunning test  .......')
    loss, acc = SNLIModel.evaluate([test_sequence[0], test_sequence[1]], test_sequence[2], batch_size=config.batch_size)
    print_log_file('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc), file=log)

    label_predicted = SNLIModel.predict([test_sequence[0], test_sequence[1]])

    Y = list() # contain labels of each sample
    Y_predicted = list()
    print('Test_seq:', len(test_sequence[2]))
    for bin_label in test_sequence[2]:
        if bin_label[0] == 1:
            Y.append(0)
        elif bin_label[1] == 1:
            Y.append(1)
        elif bin_label[2] == 1:
            Y.append(2)

    for bin_label in label_predicted:
        max_value = max(bin_label)
        if bin_label[0] == max_value:
            Y_predicted.append(0)
        if bin_label[1] == max_value:
            Y_predicted.append(1)
        if bin_label[2] == max_value:
            Y_predicted.append(2)

    label_predicted = np.asarray(Y_predicted)
    # print('Label Predicted:;', label_predicted[:20])
    print('Precision, Recall and F1-score...')
    # print(metrics.classification_report(test_sequence[2], label_predicted))
    print(metrics.classification_report(Y, Y_predicted))

    print('Confusion Matrix...')
    print(metrics.confusion_matrix(Y, Y_predicted))

    # For output an image of model
    tf.keras.utils.plot_model(
        SNLIModel, to_file='SNLI2_LSTM.png', show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    end_time = time.time()
    time_usage = timedelta(seconds=int(round(end_time - start_time)))
    print_log_file('Time usage: {0}'.format(time_usage), file=log)

    log.close()

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1, config.num_epoch+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()