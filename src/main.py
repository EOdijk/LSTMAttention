import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import cPickle as pickle
from data import Data
from model import Model
import numpy as np


def get_dataset(max_len, tuple_sentences, word_emb_size, word_emb_pretrained,
                validation_portion=0, negatives_portion=0.1, batch_size=1):
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    prep_path = root_path + '/data/prep_data_ml%d_ts%d_np%d_we%d.pk' \
                            % (max_len, tuple_sentences, negatives_portion * 100, word_emb_size)
    d = None

    if os.path.exists(prep_path):
        print 'Existing dataset found. Opening...'
        d = pickle.load(open(prep_path, 'rb'))

    if not d:
        d = Data(max_len, tuple_sentences, negatives_portion, word_emb_size, word_emb_pretrained, root_path, batch_size)
        f = open(prep_path, 'wb')
        pickle.dump(d, f, 2)
        print '  New dataset saved.'

    print 'Splitting data...'
    d.shuffle()

    d.split_validation(validation_portion)  # Performs a necessary post-split operation even if validation_portion is 0
    print_labels = True
    if print_labels:
        t_rels = {}
        v_rels = {}
        test_rels = {}
        print "Label frequencies:"
        for i in xrange(0, len(d.train)):
            r = d.train[i][2][0]
            if r in t_rels:
                t_rels[r] += 1
            else:
                t_rels[r] = 1
        for i in xrange(0, len(d.validation)):
            r = d.validation[i][2][0]
            if r in v_rels:
                v_rels[r] += 1
            else:
                v_rels[r] = 1
        for i in xrange(0, len(d.test)):
            r = d.test[i][2][0]
            if r in test_rels:
                test_rels[r] += 1
            else:
                test_rels[r] = 1
        for i in xrange(0, d.rel_count):
            t = t_rels[i] if i in t_rels else 0
            v = v_rels[i] if i in v_rels else 0
            te = test_rels[i] if i in test_rels else 0
            print i, ": ", t, "~", v, '~', te

    # if batch_size > 1:
    #     d.split_batches(batch_size)
    return d


def train(data, model, large_batch_size, epochs, early_stop, word_attention, do_test, run_name, validation_portion):
    train_batches = len(data.train)
    if validation_portion > 0.0:
        valid_batches = len(data.validation)
    test_batches = len(data.test)
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path = root_path + '/saved/' + run_name
    if os.path.exists(path + '_weights.npz'):
        model.load(path)

    all = np.arange(0, data.rel_count)
    all_relations = all
    for i in xrange(0, large_batch_size - 1):
        all_relations = np.vstack((all_relations, all))

    for ep in xrange(model.last_epoch + 1, epochs+1):
        print '  Epoch %d...' % ep
        model.last_epoch = ep
        model.train(data, train_batches, word_attention)
        if validation_portion > 0.0:
            model.evaluate(data, valid_batches, all_relations, word_attention, True, path)
            if model.epochs_since_best >= early_stop:
                print '  No more improvement seen after %d epochs, best was at %d' % (model.last_epoch, model.best_epoch)
                break

    if do_test:
        if validation_portion > 0.0 and model.best_epoch != model.last_epoch:
            model.load(path)

        print 'Testing'
        model.evaluate(data, test_batches, all_relations, word_attention, False, path)


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--epochs', metavar='epochs', nargs='?', type=int, default=1)
    parser.add_argument('--validation_portion', metavar='validation_portion', nargs='?', type=float, default=0.1,
                        help='Portion of the training data to be used for validation')
    parser.add_argument('--negatives_portion', metavar='negatives_portion', nargs='?', type=float, default=0.1,
                        help='Portion of the negatives kept in the training/validation data.')
    parser.add_argument('--batch_size', metavar='batch_size', nargs='?', type=int, default=160)
    parser.add_argument('--max_len', metavar='max_len', nargs='?', type=int, default=100,
                        help='Maximum sentence length in the dataset')
    parser.add_argument('--learning_rate', metavar='learning_rate', nargs='?', type=float, default=0.01)
    parser.add_argument('--tuple_sentences', metavar='tuple_sentences', nargs='?', type=int, default=20,
                        help='Number of sentences per tuple in a batch')
    parser.add_argument('--sentence_encoder', metavar='sentence_encoder', nargs='?', type=str, default='cnn',
                        help='The type of sentence encoder used. Can be "lstm", "cnn" or "both"')
    parser.add_argument('--units_sentence_encoder', metavar='units_sentence_encoder', nargs='?', type=int, default=20,
                        help='Number of hidden units in the LSTM / filters in the CNN')
    parser.add_argument('--cnn_filter_size', metavar='cnn_filters', nargs='?', type=int, default=3,
                        help='Filter size for the CNN. Only relevant when sentence_encoder is CNN')
    parser.add_argument('--word_attention', metavar='word_attention', nargs='?', type=int, default=0,
                        help='Whether word attention should be included or not. 1 for true, else false')
    parser.add_argument('--word_emb_size', metavar='word_emb_size', nargs='?', type=int, default=50)
    parser.add_argument('--word_emb_pretrained', metavar='word_emb_pretrained', nargs='?', default=1,
                        help='Using pre-trained word emb. Searches in data folder for "glove.6B.[word_emb_size]d.txt"')
    parser.add_argument('--pos_emb_size', metavar='pos_emb_size', nargs='?', type=int, default=3)
    parser.add_argument('--dropout_probability', metavar='dropout_probability', nargs='?', type=float, default=0.5)
    parser.add_argument('--early_stop', metavar='early_stop', nargs='?', type=int, default=20,
                        help='Number of epochs without improvement before the model stops')
    parser.add_argument('--run_name', metavar='run_name', nargs='?', type=str, default='debug',
                        help='Name of the run. Used for saving/loading.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    word_attention = True if args.word_attention == 1 else False
    word_emb_pretrained = True if args.word_emb_pretrained == 1 else False

    data = get_dataset(args.max_len, args.tuple_sentences, args.word_emb_size, word_emb_pretrained,
                       args.validation_portion, args.negatives_portion, args.batch_size)

    model = Model(args.sentence_encoder, data, args.batch_size, args.tuple_sentences,
                  args.learning_rate, args.units_sentence_encoder, args.cnn_filter_size, word_attention,
                  args.word_emb_size, args.pos_emb_size, args.dropout_probability, word_emb_pretrained)

    train(data, model, args.batch_size * args.tuple_sentences, args.epochs, args.early_stop,
          word_attention, True, args.run_name, args.validation_portion)
