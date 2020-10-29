import time
from  SNLI_Train import load_vocab, transform_data, next_batch, get_time
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn import metrics
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', dest='model', default='model/SNLI', help='The SNLI model')
    args = parser.parse_args()

    test_file = 'test.txt'
    # SNLI_model = 'model/SNLI'
    print("Loading model: " + args.model)
    start_time = time.time()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(args.model + '.meta')
    saver.restore(sess, args.model)
    graph = tf.get_default_graph()

    vocab = graph.get_tensor_by_name('vocab:0')  # file name
    mlen1 = graph.get_tensor_by_name('mlen1:0')
    mlen2 = graph.get_tensor_by_name('mlen2:0')
    vocab, mlen1, mlen2 = sess.run([vocab, mlen1, mlen2])
    print("Loading vocabulary: " + vocab.decode())
    vocab = load_vocab(vocab, adding=True)  # word to index mapping

    # get the needed placeholders and operations
    s1_ph = graph.get_tensor_by_name('sen1:0')
    s1_m_ph = graph.get_tensor_by_name('sen1_mask:0')
    s2_ph = graph.get_tensor_by_name('sen2:0')
    s2_m_ph = graph.get_tensor_by_name('sen2_mask:0')
    dropout_keep_ph = graph.get_tensor_by_name('dropout_keep:0')
    pred_op = graph.get_tensor_by_name('predict/y:0')

    # load test data
    print("Loading test data: " + test_file)
    s1, s1_mask, s2, s2_mask, y = \
        transform_data(test_file, vocab, max_len1=mlen1, max_len2=mlen2, lowercase=True)
    batches = next_batch(s1, s1_mask, s2, s2_mask, y, shuffle=False)

    data_len = len(s1)
    y_pred = []

    # predict
    print('Predicting on', test_file, '.....')
    for batch in batches:
        s1_batch, s1_batch_mask, s2_batch, s2_batch_mask, _ = batch
        feed_dict = {
            s1_ph: s1_batch,
            s1_m_ph: s1_batch_mask,
            s2_ph: s2_batch,
            s2_m_ph: s2_batch_mask,
            dropout_keep_ph: 1.0,
        }
        pred = sess.run(pred_op, feed_dict)
        y_pred.extend(pred)

    y_pred = np.asarray(y_pred, dtype=np.int32)

    accuracy = np.mean(y == y_pred)
    print('Accuracy: {0:>6.2%}\n'.format(accuracy))

    print('Precision, Recall and F1-score...')
    print(metrics.classification_report(y, y_pred))

    print('Confusion Matrix...')
    print(metrics.confusion_matrix(y, y_pred))

    time_dif = get_time(start_time)
    print('Time usage:', time_dif)
