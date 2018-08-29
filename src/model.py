import os
import numpy as np
import lasagne
import theano
import theano.tensor as T
import cPickle as pickle
import math
from sklearn import metrics

"""
Bottom-up overview:
x_i = lstm output
e_ir = x_i * A * r, indication of how well the label r fits the mention x_i
alpha_ir = softmax over e, weight of sentence x_i in the attention model
s_r = sum over alpha_ir * x_i
O_r = M * s_r + d, output of the model
p(r|S,theta) = softmax over O, the conditional probability of a label r given an entity's mentions S
"""


class ExpressionMergeLayer(lasagne.layers.MergeLayer):
    """
    Custom layer. Source: https://github.com/Lasagne/Lasagne/issues/584

    This layer performs an custom expressions on list of inputs to merge them.
    This layer is different from ElemwiseMergeLayer by not required all
    input_shapes are equal

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes

    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.

    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.

    Notes
    -----
    if ``output_shape=None``, this layer chooses first input_shape as its
    output_shape
    """

    def __init__(self, incomings, merge_function, output_shape=None, **kwargs):
        super(ExpressionMergeLayer, self).__init__(incomings, **kwargs)
        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        if self._output_shape is None:
            return input_shapes[0]
        elif self._output_shape is 'auto':
            input_shape = [(0 if s is None else s for s in ishape)
                           for ishape in input_shapes]
            Xs = [T.alloc(0, *ishape) for ishape in input_shape]
            output_shape = self.merge_function(*Xs).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, inputs, **kwargs):
        return self.merge_function(*inputs)


class SentenceLabeler(lasagne.layers.MergeLayer):
    """
    e_ir = x_i * A * r
    Custom Lasagne layer used by Model class
    """

    def __init__(self, x_i, r, batch_size, tuple_sentences, rel_count, emb_size,
                 test, word_attention, A=lasagne.init.Normal(), **kwargs):
        super(SentenceLabeler, self).__init__([x_i, r], **kwargs)
        self.A = self.add_param(A, (emb_size, emb_size), name='A')
        self.test = test
        self.word_attention = word_attention

        self.batch_size = batch_size
        self.tuple_sentences = tuple_sentences
        self.rel_count = rel_count
        self.emb_size = emb_size

    def get_output_for(self, inputs, **kwargs):

        if self.test:
            if self.word_attention:

                # [B,2h,|L|]->[B,|L|,2h]
                x_i_shuffle = inputs[0].dimshuffle(0, 2, 1)

                # [B,|L|,2h][2h,2h]
                first_dot = T.tensordot(x_i_shuffle, self.A, axes=[2, 0])

                # [B,|L|,2h]->[B*|L|,2h] for both
                first_dot_reshape = T.reshape(first_dot,
                                              (self.batch_size * self.tuple_sentences * self.rel_count, self.emb_size))
                r_reshape = T.reshape(inputs[1].dimshuffle(0, 2, 1),
                                      (self.batch_size * self.tuple_sentences * self.rel_count, self.emb_size))

                # [B*|L|,2h][B*|L|,2h] batched
                second_dot = T.batched_dot(first_dot_reshape, r_reshape)

                # [B*|L|]->[B,|L|]
                second_dot_reshape = T.reshape(second_dot, (self.batch_size * self.tuple_sentences, self.rel_count))

                return second_dot_reshape

            else:
                # [B,2h][2h,2h]
                # [B,2h][B,2h,|L|] batched
                return T.batched_dot(T.tensordot(inputs[0], self.A, axes=[1, 0]), inputs[1].dimshuffle(0, 2, 1))
        else:
            # [B,2h][2h,2h]
            # [B,2h][B,2h] batched
            return T.batched_dot(T.tensordot(inputs[0], self.A, axes=[1, 0]), inputs[1])

    def get_output_shape_for(self, input_shapes):
        if self.test:
            return input_shapes[0][0], input_shapes[1][1]  # [B,|L|]
        else:
            return input_shapes[0][0],  # [B]


class WordLabeler(lasagne.layers.MergeLayer):
    """
    e_ir = x_i * r
    Custom layer. Like SentenceLabeler, but without a matrix in between
    """

    def __init__(self, x_i, r, test, **kwargs):
        super(WordLabeler, self).__init__([x_i, r], **kwargs)
        self.test = test

    def get_output_for(self, inputs, **kwargs):

        if self.test:
            # [B*t,(2)h][B*t,2h,|L|] batched
            return T.batched_dot(inputs[0], inputs[1].dimshuffle(0, 2, 1))
        else:
            # [B*t,(2)h][B*t,(2)h] batched
            return T.batched_dot(inputs[0], inputs[1])

    def get_output_shape_for(self, input_shapes):
        if self.test:
            return input_shapes[0][0], input_shapes[1][1]  # [B*t,|L|]
        else:
            return input_shapes[0][0],  # [B*t]


class WeightedSequence(lasagne.layers.MergeLayer):
    """
    For sentences: s_r = sum over alpha_ir * x_i
    For words: w_r = sum over alpha_ir * x_i (B instead of b, t instead of S)
    Custom layer to get a weighted sum over a sequence of words (for word attention) and sentences (for sentence att.)
    """

    def __init__(self, x_i, alpha_ir, test, has_L, **kwargs):
        super(WeightedSequence, self).__init__([x_i, alpha_ir], **kwargs)
        self.test = test
        self.has_L = has_L  # Whether input[0] has an |L| dimension

    def get_output_for(self, inputs, **kwargs):
        if self.test:
            if self.has_L:
                # elem-wise mult [b,S,2h,|L|][b,S,1,|L|]
                return T.sum(inputs[0] * inputs[1].dimshuffle(0, 1, 'x', 2), axis=1)  # [b,2h,|L|]
            else:
                # elem-wise mult [b,S,2h,1][b,S,1,|L|], sum over S
                return T.sum(inputs[0].dimshuffle(0, 1, 2, 'x') * inputs[1].dimshuffle(0, 1, 'x', 2), axis=1)  # [b,2h,|L|]
        else:
            # elem-wise mult [b,S,2h][b,S,1], sum over S
            return T.sum(inputs[0] * inputs[1].dimshuffle(0, 1, 'x'), axis=1)

    def get_output_shape_for(self, input_shapes):
        if self.test:
            return input_shapes[0][0], input_shapes[0][2], input_shapes[1][2]  # [b,2h,|L|]
        else:
            return input_shapes[0][0], input_shapes[0][2]  # [b,2h]


class ModelStats:
    """Lightweight class containing the Model's stats. Used for serialization."""

    def __init__(self, last_epoch, best_epoch, best_pr_area, epochs_since_best,
                 precision_list, recall_list, f1_measure_list, pr_area_list,
                 learning_rate, sentence_encoder, units_sentence_encoder, cnn_filter_size,
                 word_emb_size, pos_emb_size, dropout_prob,
                 precision_test, recall_test, f1_measure_test, pr_area_test,
                 predictions_val, predictions_test, truths_val, truths_test):
        self.last_epoch = last_epoch
        self.best_epoch = best_epoch
        self.best_pr_area = best_pr_area
        self.epochs_since_best = epochs_since_best
        self.precision_list = precision_list
        self.recall_list = recall_list
        self.f1_measure_list = f1_measure_list
        self.pr_area_list = pr_area_list
        self.precision_test = precision_test
        self.recall_test = recall_test
        self.f1_measure_test = f1_measure_test
        self.pr_area_test = pr_area_test
        self.learning_rate = learning_rate
        self.sentence_encoder = sentence_encoder
        self.units_sentence_encoder = units_sentence_encoder
        self.cnn_filter_size = cnn_filter_size
        self.word_emb_size = word_emb_size
        self.pos_emb_size = pos_emb_size
        self.dropout_prob = dropout_prob
        self.predictions_val = predictions_val
        self.predictions_test = predictions_test
        self.truths_val = truths_val
        self.truths_test = truths_test


class Model:
    """
    Main class to wrap the model together.
    Several steps are wrapped in separate 'create_' functions
    """

    def __init__(self, sentence_encoder, data, batch_size, tuple_sentences, learning_rate,
                 units_sentence_encoder, cnn_filter_size, word_attention,
                 word_emb_size, pos_emb_size, dropout_probability, word_emb_pretrained):
        self.last_epoch = 0
        self.best_epoch = 0
        self.best_pr_area = 0
        self.epochs_since_best = 0

        self.precision_list = []
        self.recall_list = []
        self.f1_measure_list = []
        self.pr_area_list = []

        self.predictions_val = []
        self.truths_val = []

        self.precision_test = []
        self.recall_test = []
        self.f1_measure_test = []
        self.pr_area_test = []

        self.predictions_test = []
        self.truths_test = []

        self.learning_rate = learning_rate
        self.sentence_encoder = sentence_encoder
        self.units_sentence_encoder = units_sentence_encoder
        self.cnn_filter_size = cnn_filter_size
        self.word_attention = word_attention
        self.word_emb_size = word_emb_size
        self.pos_emb_size = pos_emb_size
        self.dropout_prob = dropout_probability

        large_batch = batch_size * tuple_sentences
        target_values = T.ivector('target_output')
        if sentence_encoder.lower() == 'lstm':
            post_encoder_units = 2 * units_sentence_encoder  # doubled due to bi-lstm
        else:
            post_encoder_units = units_sentence_encoder

        print "Building Model..."

        """-----1. Inputs-----"""
        l_in = lasagne.layers.InputLayer(shape=(large_batch, data.max_len), input_var=T.imatrix())  # [B,t]
        l_mask = lasagne.layers.InputLayer(shape=(large_batch, data.max_len))
        l_in_pos1 = lasagne.layers.InputLayer(shape=(large_batch, data.max_len), input_var=T.imatrix())  # [B,t]
        l_in_pos2 = lasagne.layers.InputLayer(shape=(large_batch, data.max_len), input_var=T.imatrix())  # [B,t]
        l_r_in = lasagne.layers.InputLayer(shape=(large_batch,), input_var=T.ivector())  # [B]
        l_r_in_test = lasagne.layers.InputLayer(shape=(large_batch, data.rel_count),
                                                input_var=T.imatrix())  # [B,|L|]
        l_r_words_in = lasagne.layers.InputLayer(shape=(large_batch*data.max_len,), input_var=T.ivector())  # [B*t]
        l_r_words_in_test = lasagne.layers.InputLayer(shape=(large_batch*data.max_len, data.rel_count),
                                                      input_var=T.imatrix())  # [B*t,|L|]

        """-----2. Embeddings-----"""
        l_emb = self.create_input_embeddings(l_in, l_in_pos1, l_in_pos2, data.max_len, data.word_count,
                                             word_emb_size, pos_emb_size, data, word_emb_pretrained)
        l_r_emb = lasagne.layers.EmbeddingLayer(l_r_in, data.rel_count, post_encoder_units)  # [B,2h]
        l_r_emb_test = lasagne.layers.EmbeddingLayer(l_r_in_test, data.rel_count,
                                                     post_encoder_units, W=l_r_emb.W)  # [B,|L|,2h]
        l_r_words_emb = lasagne.layers.EmbeddingLayer(l_r_words_in, data.rel_count, post_encoder_units)  # [B*t,2h]
        l_r_words_emb_test = lasagne.layers.EmbeddingLayer(l_r_words_in_test, data.rel_count,
                                                           post_encoder_units, W=l_r_words_emb.W)  # [B*t,|L|,2h]

        """-----3. Encoder-----"""
        x_i = self.create_sentence_encoder(l_emb, l_mask, units_sentence_encoder, sentence_encoder, cnn_filter_size,
                                           word_attention)
        if word_attention:
            # Note: called "reshape" for consistency with the non-word_attention version.
            #       Reshape has the word or sentence dimension separate from the rest
            x_reshape = x_i
            # [B,t,(2)h]->[B*t,(2)h]
            x_i = lasagne.layers.ReshapeLayer(x_i, (batch_size*tuple_sentences*data.max_len, post_encoder_units))
        else:
            # [B,(2)h]->[b,S,(2)h]
            x_reshape = lasagne.layers.ReshapeLayer(x_i, (batch_size, tuple_sentences, post_encoder_units))

        """-----4a. Training branch-----"""
        cost, updates = self.create_training_branch(l_r_emb, l_r_words_emb,
                                                    x_i, x_reshape, target_values, data.rel_count,
                                                    post_encoder_units, batch_size, tuple_sentences,
                                                    dropout_probability, learning_rate, word_attention,
                                                    data, post_encoder_units)
        if word_attention:
            self.train_function = theano.function([l_in.input_var, l_in_pos1.input_var, l_in_pos2.input_var,
                                                  l_mask.input_var, l_r_in.input_var, l_r_words_in.input_var,
                                                   target_values],
                                                  cost,
                                                  updates=updates,
                                                  allow_input_downcast=True)
        else:
            self.train_function = theano.function([l_in.input_var, l_in_pos1.input_var, l_in_pos2.input_var,
                                                  l_mask.input_var, l_r_in.input_var, target_values],
                                                  cost,
                                                  updates=updates,
                                                  allow_input_downcast=True)

        """-----4b. Testing branch-----"""
        outputs = self.create_testing_branch(l_r_emb_test, l_r_words_emb_test,
                                             x_i, x_reshape, data.rel_count, post_encoder_units,
                                             batch_size, tuple_sentences, dropout_probability, word_attention,
                                             data, post_encoder_units)
        if word_attention:
            self.compute_outputs = theano.function([l_in.input_var, l_in_pos1.input_var, l_in_pos2.input_var,
                                                    l_mask.input_var, l_r_in_test.input_var,
                                                    l_r_words_in_test.input_var],
                                                    outputs, allow_input_downcast=True)
        else:
            self.compute_outputs = theano.function([l_in.input_var, l_in_pos1.input_var, l_in_pos2.input_var,
                                                    l_mask.input_var, l_r_in_test.input_var],
                                                   outputs, allow_input_downcast=True)

    def create_input_embeddings(self, l_in, l_in_pos1, l_in_pos2, max_len, word_count, word_emb_size, pos_emb_size,
                                data, word_emb_pretrained):
        """Creates the layers for word embeddings"""

        # Create embeddings
        if word_emb_pretrained:
            # [B,t,h_w]
            l_emb = lasagne.layers.EmbeddingLayer(l_in, word_count, word_emb_size, W=data.pretrained_embeddings)
            l_emb.params[l_emb.W].remove('trainable')
        else:
            l_emb = lasagne.layers.EmbeddingLayer(l_in, word_count, word_emb_size)  # [B,t,h_w]
        l_emb_pos1 = lasagne.layers.EmbeddingLayer(l_in_pos1, max_len, pos_emb_size)  # [B,t,h_e]
        l_emb_pos2 = lasagne.layers.EmbeddingLayer(l_in_pos2, max_len, pos_emb_size)  # [B,t,h_e]

        # Concatenate embeddings
        l_emb_concat = lasagne.layers.ConcatLayer([l_emb, l_emb_pos1, l_emb_pos2], axis=2)  # [B,t,h = h_w + 2h_e]

        return l_emb_concat

    def create_lstm(self, l_emb, l_mask, units, word_attention):
        """Bi-directional LSTM, masked."""

        # LSTM
        l_fwd = lasagne.layers.LSTMLayer(l_emb, units, mask_input=l_mask, grad_clipping=10,
                                         nonlinearity=lasagne.nonlinearities.tanh, backwards=False, name='l_fwd')
        l_bwd = lasagne.layers.LSTMLayer(l_emb, units, mask_input=l_mask, grad_clipping=10,
                                         nonlinearity=lasagne.nonlinearities.tanh, backwards=True, name='l_bwd')
        l_out = lasagne.layers.ConcatLayer([l_fwd, l_bwd], axis=2)  # [B,t,2h]
        l_out = lasagne.layers.DimshuffleLayer(l_out, (0, 2, 1))  # [B,t,2h]->[B,2h,t]

        # Apply mask
        l_mask = lasagne.layers.DimshuffleLayer(l_mask, (0, 'x', 1))  # [B,t]->[B,x,t]
        l_out = ExpressionMergeLayer([l_out, l_mask], theano.tensor.mul)  # [B,2h,t]

        if word_attention:
            l_out = lasagne.layers.DimshuffleLayer(l_out, (0, 2, 1))  # [B,2h,t]->[B,t,2h]
        else:
            l_out = lasagne.layers.GlobalPoolLayer(l_out, pool_function=T.max)  # [B,2h]

        return l_out

    def create_cnn(self, l_emb, l_mask, units, cnn_filter_size, word_attention):
        """CNN, masked."""

        # CNN
        l_emb_shuffle = lasagne.layers.DimshuffleLayer(l_emb, (0, 2, 1))  # [B,t,h]->[B,h,t]
        x_cnn = lasagne.layers.Conv1DLayer(l_emb_shuffle, units, cnn_filter_size, pad='same')  # [B,h,t]

        # Apply mask
        l_mask = lasagne.layers.DimshuffleLayer(l_mask, (0, 'x', 1))  # [B,t]->[B,x,t]
        x_cnn = ExpressionMergeLayer([x_cnn, l_mask], theano.tensor.mul)  # [B,h,t]

        if word_attention:
            l_out = lasagne.layers.DimshuffleLayer(x_cnn, (0, 2, 1))  # [B,h,t]->[B,t,h]
        else:
            l_out = lasagne.layers.GlobalPoolLayer(x_cnn, pool_function=T.max)  # [B,h]

        return l_out

    def create_sentence_encoder(self, l_emb, l_mask, units, sentence_encoder, cnn_filter_size, word_attention):
        """Creates the layers for encoding, using create_lstm and create_cnn."""

        if sentence_encoder.lower() == 'lstm':
            print('Encoding sentences with BiLSTM')
            x_i = self.create_lstm(l_emb, l_mask, units, word_attention)

        elif sentence_encoder.lower() == 'cnn':
            print('Encoding sentences with CNN')
            x_i = self.create_cnn(l_emb, l_mask, units, cnn_filter_size, word_attention)

        # Both
        else:
            print('Encoding sentences with BiLSTM+CNN')
            l_lstm = self.create_lstm(l_emb, l_mask, units, False)
            x_i = self.create_cnn(l_lstm, l_mask, units, cnn_filter_size, word_attention)

        return x_i

    def create_training_branch(self, l_r_emb, l_r_words_emb,
                               x_i, x_reshape, target_values, rel_count, units_sentence_encoder,
                               batch_size, tuple_sentences, dropout_probability, learning_rate, word_attention,
                               data, post_encoder_units):
        """Creates the cost and output for the training branch"""

        if word_attention:
            # [B,t,(2)h]->[B*t,(2)h]
            #x_bt_reshape = lasagne.layers.ReshapeLayer(x_i, (batch_size * tuple_sentences * data.max_len,
            #                                                 post_encoder_units))

            # c_ir
            l_c_ir = WordLabeler(x_i, l_r_words_emb, False)  # [B*t]
            c_reshape = lasagne.layers.ReshapeLayer(l_c_ir, (batch_size*tuple_sentences, data.max_len))  # [B*t]->[B,t]

            # beta_ir
            l_beta_ir = lasagne.layers.NonlinearityLayer(c_reshape,
                                                         nonlinearity=lasagne.nonlinearities.softmax)  # [B,t]

            # w_r
            l_w_r = WeightedSequence(x_reshape, l_beta_ir, False, False)  # [B,(2)h]

            x_i_sentence = l_w_r  # [B,(2)h]
            x_reshape_sentence = lasagne.layers.ReshapeLayer(x_i_sentence, (batch_size, tuple_sentences,
                                                                            post_encoder_units))  # [B,(2)h]->[b,S,(2)h]
        else:
            x_i_sentence = x_i  # [B,(2)h]
            x_reshape_sentence = x_reshape  # [b,S,(2)h]

        # e_ir
        l_e_ir = SentenceLabeler(x_i_sentence, l_r_emb, batch_size, tuple_sentences, rel_count,
                                 units_sentence_encoder, False, word_attention)  # [B]
        e_reshape = lasagne.layers.ReshapeLayer(l_e_ir, (batch_size, tuple_sentences))  # [B]->[b,S]

        # alpha_ir
        l_alpha_ir = lasagne.layers.NonlinearityLayer(e_reshape,
                                                      nonlinearity=lasagne.nonlinearities.softmax)  # [b,S]

        # s_r
        l_s_r = WeightedSequence(x_reshape_sentence, l_alpha_ir, False, word_attention)  # [b,(2)h]

        # Dropout
        l_dropout = lasagne.layers.DropoutLayer(l_s_r, dropout_probability)  # [b,(2)h]

        # Softmax over O_r
        self.l_out = lasagne.layers.DenseLayer(l_dropout, num_units=rel_count,
                                               nonlinearity=lasagne.nonlinearities.softmax)  # [b,|L|]

        # Final
        network_output = lasagne.layers.get_output(self.l_out, deterministic=False)
        cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
        updates = lasagne.updates.adam(cost,
                                          lasagne.layers.get_all_params(self.l_out, trainable=True),
                                          learning_rate)

        return cost, updates

    def create_testing_branch(self, l_r_emb_test, l_r_words_emb_test, x_i, x_reshape,
                              rel_count, units_sentence_encoder,
                              batch_size, tuple_sentences, dropout_probability, word_attention,
                              data, post_encoder_units):
        """Creates the output for the testing branch"""

        if word_attention:
            # [B,t,(2)h]->[B*t,(2)h]
            #x_bt_reshape = lasagne.layers.ReshapeLayer(x_i, (batch_size * tuple_sentences * data.max_len,
            #                                                 post_encoder_units))

            # c_ir
            l_c_ir = WordLabeler(x_i, l_r_words_emb_test, True)  # [B*t,|L|]

            # [B*t,|L|]->[B,t,|L|]
            c_reshape = lasagne.layers.ReshapeLayer(l_c_ir, (batch_size*tuple_sentences, data.max_len, rel_count))
            # [B,t,|L|]->[B,|L|,t]
            c_reshape = lasagne.layers.DimshuffleLayer(c_reshape, (0, 2, 1))
            # [B,|L|,t]->[B*|L|,t]
            c_reshape = lasagne.layers.ReshapeLayer(c_reshape, (batch_size*tuple_sentences*rel_count, data.max_len))

            # beta_ir
            l_beta_ir = lasagne.layers.NonlinearityLayer(c_reshape,
                                                         nonlinearity=lasagne.nonlinearities.softmax)  # [B*|L|,t]

            # [B*|L|,t]->[B,|L|,t]
            l_beta_ir_reshape = lasagne.layers.ReshapeLayer(l_beta_ir,
                                                            (batch_size*tuple_sentences, rel_count, data.max_len))
            # [B,|L|,t]->[B,t,|L|]
            l_beta_ir_reshape = lasagne.layers.DimshuffleLayer(l_beta_ir_reshape, (0, 2, 1))

            # w_r
            l_w_r = WeightedSequence(x_reshape, l_beta_ir_reshape, True, False)  # [B,(2)h,|L|]

            x_i_sentence = l_w_r  # [B,(2)h,|L|]
            # [B,(2)h,|L|]->[b,S,(2)h,|L|]
            x_reshape_sentence = lasagne.layers.ReshapeLayer(
                x_i_sentence, (batch_size, tuple_sentences, post_encoder_units, rel_count))
        else:
            x_i_sentence = x_i  # [B,(2)h]
            x_reshape_sentence = x_reshape  # [b,S,(2)h]

        # e_ir
        # [B,|L|]
        l_e_ir_test = SentenceLabeler(x_i_sentence, l_r_emb_test, batch_size, tuple_sentences, rel_count,
                                      units_sentence_encoder, True, word_attention)
        # [B,|L|]->[b,S,|L|]
        e_reshape_test = lasagne.layers.ReshapeLayer(l_e_ir_test, (batch_size, tuple_sentences, rel_count))
        # [b,S,|L|]->[b,|L|,S]
        e_reshape_test = lasagne.layers.DimshuffleLayer(e_reshape_test, (0, 2, 1))
        # [b,|L|,S]->[b*|L|,S]
        e_reshape_test = lasagne.layers.ReshapeLayer(e_reshape_test, (batch_size * rel_count, tuple_sentences))

        # alpha_ir
        l_alpha_ir_test = lasagne.layers.NonlinearityLayer(e_reshape_test,
                                                           nonlinearity=lasagne.nonlinearities.softmax)  # [b*|L|,S]
        # [b*|L|,S]->[b,|L|,S]
        l_alpha_reshape_test = lasagne.layers.ReshapeLayer(l_alpha_ir_test,
                                                           (batch_size, rel_count, tuple_sentences))
        # [b,|L|,S]->[b,S,|L|]
        l_alpha_reshape_test = lasagne.layers.DimshuffleLayer(l_alpha_reshape_test, (0, 2, 1))

        # s_r
        l_s_r_test = WeightedSequence(x_reshape_sentence, l_alpha_reshape_test, True, word_attention)  # [b,(2)h]

        # Dropout
        l_dropout_test = lasagne.layers.DropoutLayer(l_s_r_test, dropout_probability)  # [b,(2)h,|L|]

        # [b,(2)h,|L|]->[b,|L|,(2)h]
        l_dropout_shuffle_test = lasagne.layers.DimshuffleLayer(l_dropout_test, (0, 2, 1))

        # [b,|L|,(2h)]->[b*|L|,(2)h]
        l_dropout_reshape_test = lasagne.layers.ReshapeLayer(l_dropout_shuffle_test,
                                                             (batch_size*rel_count, units_sentence_encoder))

        # Softmax over O_r
        # Note: first |L| dimension is for the different s_r, second |L| dimension is their results
        l_out_test_pre_reshape = lasagne.layers.DenseLayer(l_dropout_reshape_test, num_units=rel_count,
                                                    W=self.l_out.W, b=self.l_out.b,
                                                    nonlinearity=lasagne.nonlinearities.softmax
                                                    # ,
                                                    # num_leading_axes=-1
                                                    )  # [b*|L|,|L|]

        # [b*|L|,|L|]->[b,|L|,|L|]
        self.l_out_test = lasagne.layers.ReshapeLayer(l_out_test_pre_reshape, (batch_size, rel_count, rel_count))

        # Final
        network_output_test = lasagne.layers.get_output(self.l_out_test, deterministic=True)
        # outputs = T.max_and_argmax(network_output_test, axis=1)

        return network_output_test

    def train(self, data, train_batches, word_attention):
        """Main training function."""

        print '  Training...'
        d = 0
        while d < len(data.train):
            # print d
            if d + data.batch_size >= len(data.train):
                train_batch = data.train[d:len(data.train)]
                new_tr_batch = []
                for ex in train_batch:
                    one = data._pad_matrix(ex[0])
                    two = data._pad_matrix(ex[1])
                    three = ex[2]
                    four = ex[3]
                    five = data._pad_matrix(ex[4])
                    six = data._pad_matrix(ex[5])
                    seven = data._pad_matrix(ex[6])
                    eight = data._pad_matrix(ex[7])
                    new_tr_batch.append([one, two, three, four, five, six, seven, eight])
                for i in xrange(data.batch_size - len(train_batch)):
                    new_tr_batch.append([np.zeros_like(new_tr_batch[0][0]), np.zeros_like(new_tr_batch[0][1]),
                                         np.zeros_like(new_tr_batch[0][2]), np.zeros_like(new_tr_batch[0][3]),
                                         np.zeros_like(new_tr_batch[0][4]), np.zeros_like(new_tr_batch[0][5]),
                                         np.zeros_like(new_tr_batch[0][6]), np.zeros_like(new_tr_batch[0][7])])

            else:
                train_batch = data.train[d:d + data.batch_size]
                new_tr_batch =[]
                for ex in train_batch:
                    one = data._pad_matrix(ex[0])
                    two = data._pad_matrix(ex[1])
                    three = ex[2]
                    four =  ex[3]
                    five = data._pad_matrix(ex[4])
                    six = data._pad_matrix(ex[5])
                    seven = data._pad_matrix(ex[6])
                    eight = data._pad_matrix(ex[7])
                    new_tr_batch.append([one, two, three, four, five, six, seven, eight])

                current_batch = data.split_list(new_tr_batch, data.batch_size)
                current_batch =current_batch[0]

            d += data.batch_size

        # for batch in xrange(0, train_batches):
        #     if batch % 50 == 0  and batch!=0:
        #         print '    batch %d of %d...' % (batch + 1, len(data.train))


            mentions = current_batch[0]
            if self.sentence_encoder.lower() == 'lstm':
                pos1 = current_batch[7]
                pos2 = current_batch[8]
            else:
                pos1 = current_batch[4]
                pos2 = current_batch[5]


            relations_sentences = current_batch[2]
            relation_batch = current_batch[6]
            if word_attention:
                relations_words = np.repeat(relations_sentences, data.max_len)
                masks = current_batch[1]
                self.train_function(mentions, pos1, pos2, masks, relations_sentences, relations_words,
                                    relation_batch)

            else:
                masks = current_batch[1]
                self.train_function(mentions, pos1, pos2, masks, relations_sentences, relation_batch)

    def get_max_argmax(self, model_out):
        """
        Gets max and argmax from the model output (non-Theano).
        First finds the output for each projected s_r.
        """

        max_list = []
        argmax_list = []

        # For each entry in the batch
        for b in xrange(0, len(model_out)):
            max = -1
            argmax = -1
            sum = 0

            # For each s_r projection, get the score of that relation only. If best, save
            for l in xrange(0, len(model_out[b])):
                out = math.exp(model_out[b][l][l])
                sum += out
                if out > max:
                    max = out
                    argmax = l

            # Normalize
            max /= sum

            max_list.append(max)
            argmax_list.append(argmax)

        return np.array([max_list, argmax_list])

    def evaluate(self, data, num_batches, all_relations, word_attention, is_validation, path):
        """
        Computes precision, recall, f1 and p/r curve.
        all_relations refers to a matrix containing a vector of all possible relations for each entry in the large batch
        """

        print '  Evaluating...'
        all_relations_words = np.tile(all_relations, (data.max_len, 1))

        # Process batches
        # for batch in xrange(0, num_batches):
        #     if batch%50 == 0 and batch!=0:
        #         print '    Processing batch ', batch
        batch = 0
        d = 0

        if is_validation:
            while d < len(data.validation):
                if d + data.batch_size >= len(data.validation):
                    train_batch = data.validation[d:len(data.validation)]
                    new_tr_batch = []
                    for ex in train_batch:
                        one = data._pad_matrix(ex[0])
                        two = data._pad_matrix(ex[1])
                        three = ex[2]
                        four = ex[3]
                        five = data._pad_matrix(ex[4])
                        six = data._pad_matrix(ex[5])
                        seven = data._pad_matrix(ex[6])
                        eight = data._pad_matrix(ex[7])
                        new_tr_batch.append([one, two, three, four, five, six, seven, eight])
                    for i in xrange(data.batch_size - len(train_batch)):
                        new_tr_batch.append([np.zeros_like(new_tr_batch[0][0]), np.zeros_like(new_tr_batch[0][1]),
                                             np.zeros_like(new_tr_batch[0][2]), np.zeros_like(new_tr_batch[0][3]),
                                             np.zeros_like(new_tr_batch[0][4]), np.zeros_like(new_tr_batch[0][5]),
                                             np.zeros_like(new_tr_batch[0][6]), np.zeros_like(new_tr_batch[0][7])])

                else:
                    train_batch = data.validation[d:d + data.batch_size]
                    new_tr_batch = []
                    for ex in train_batch:
                        one = data._pad_matrix(ex[0])
                        two = data._pad_matrix(ex[1])
                        three = ex[2]
                        four = ex[3]
                        five = data._pad_matrix(ex[4])
                        six = data._pad_matrix(ex[5])
                        seven = data._pad_matrix(ex[6])
                        eight = data._pad_matrix(ex[7])
                        new_tr_batch.append([one, two, three, four, five, six, seven, eight])

                    current_batch = data.split_list(new_tr_batch, data.batch_size)
                    current_batch = current_batch[0]
                    mentions = current_batch[0]
                    masks = current_batch[1]
                    if self.sentence_encoder.lower() == 'lstm':
                        pos1 = current_batch[7]
                        pos2 = current_batch[8]
                    else:
                        pos1 = current_batch[4]
                        pos2 = current_batch[5]
                    relation_batch = current_batch[6]

                d += data.batch_size
                batch+=1

                if word_attention:
                    best = self.get_max_argmax(
                        self.compute_outputs(mentions, pos1, pos2, masks, all_relations, all_relations_words))
                else:
                    best = self.get_max_argmax(
                        self.compute_outputs(mentions, pos1, pos2, masks, all_relations))

                if batch == 1:  # Note: this can be prettier, numpy is picky
                    outputs = best
                else:
                    outputs = np.hstack((outputs, best))

                if batch == 1:
                    gold = relation_batch
                else:
                    gold = np.hstack((gold, relation_batch))
        else:
            while d < len(data.test):
                if d + data.batch_size >= len(data.test):
                    train_batch = data.test[d:len(data.test)]
                    new_tr_batch = []
                    for ex in train_batch:
                        one = data._pad_matrix(ex[0])
                        two = data._pad_matrix(ex[1])
                        three = ex[2]
                        four = ex[3]
                        five = data._pad_matrix(ex[4])
                        six = data._pad_matrix(ex[5])
                        seven = data._pad_matrix(ex[6])
                        eight = data._pad_matrix(ex[7])
                        new_tr_batch.append([one, two, three, four, five, six, seven, eight])
                    for i in xrange(data.batch_size - len(train_batch)):
                        new_tr_batch.append([np.zeros_like(new_tr_batch[0][0]), np.zeros_like(new_tr_batch[0][1]),
                                             np.zeros_like(new_tr_batch[0][2]), np.zeros_like(new_tr_batch[0][3]),
                                             np.zeros_like(new_tr_batch[0][4]), np.zeros_like(new_tr_batch[0][5]),
                                             np.zeros_like(new_tr_batch[0][6]), np.zeros_like(new_tr_batch[0][7])])
                    break
                else:
                    train_batch = data.test[d:d + data.batch_size]
                    new_tr_batch = []
                    for ex in train_batch:
                        one = data._pad_matrix(ex[0])
                        two = data._pad_matrix(ex[1])
                        three = ex[2]
                        four = ex[3]
                        five = data._pad_matrix(ex[4])
                        six = data._pad_matrix(ex[5])
                        seven = data._pad_matrix(ex[6])
                        eight = data._pad_matrix(ex[7])
                        new_tr_batch.append([one, two, three, four, five, six, seven, eight])

                    current_batch = data.split_list(new_tr_batch, data.batch_size)
                    current_batch = current_batch[0]
                    mentions = current_batch[0]
                    masks = current_batch[1]
                    if self.sentence_encoder.lower() == 'lstm':
                        pos1 = current_batch[7]
                        pos2 = current_batch[8]
                    else:
                        pos1 = current_batch[4]
                        pos2 = current_batch[5]
                    relation_batch = current_batch[6]

                d += data.batch_size
                batch+=1

                if word_attention:
                    best = self.get_max_argmax(
                        self.compute_outputs(mentions, pos1, pos2, masks, all_relations, all_relations_words))
                else:
                    best = self.get_max_argmax(
                        self.compute_outputs(mentions, pos1, pos2, masks, all_relations))

                if batch == 1:  # Note: this can be prettier, numpy is picky
                    outputs = best
                else:
                    outputs = np.hstack((outputs, best))

                if batch == 1:
                    gold = relation_batch
                else:
                    gold = np.hstack((gold, relation_batch))

        # print "GOLD: ", gold
        # print "OUTPUTS: ", outputs

        none_index = data._rel_dict['NA']  # Should be 0, though
        precision = []
        recall = []
        f1_measure = []
        predictions = []
        truths = []
        true_pos = 0.  # Number of correct predictions
        total_gold_pos = 0.  # Total number of non-negative golds
        total_output_pos = 0.  # Total number of non-negative outputs

        # Find total number of gold positives beforehand
        for i in xrange(0, len(gold)):
            if gold[i] != none_index:
                total_gold_pos += 1

        print "all golds: ", len(gold)
        print "non_zero golds: ", total_gold_pos

        # Loop through outputs sorted by probability, most likely predictions at the top
        # Note: outputs[0] are probabilities, outputs[1] are labels
        for i in np.array(np.argsort(outputs[0])[::-1]):

            # Save raw evaluation data
            predictions.append(outputs[1][i])
            truths.append(gold[i])

            # Find total positive outputs so far
            if int(outputs[1][i]) != none_index:
                total_output_pos += 1

            # Find true positives so far
            if gold[i] != none_index:
                if int(outputs[1][i]) == gold[i]:
                    true_pos += 1.

            # Precision
            if total_output_pos == 0:
                p = 0.
            else:
                p = float(true_pos) / float(total_output_pos)

            # Recall
            if total_gold_pos == 0:
                r = 0.
            else:
                r = float(true_pos) / float(total_gold_pos)

            # F1
            if p + r == 0.:
                f = 0.
            else:
                f = (2. * p * r) / (p + r)

            precision.append(p)
            recall.append(r)
            f1_measure.append(f)

        print "True pos: ", true_pos
        print "Total gold pos: ", total_gold_pos
        print "Total: ",

        # Save validation results
        if is_validation:
            self.precision_list.append(precision)
            self.recall_list.append(recall)
            self.f1_measure_list.append(f1_measure)

            pr_area = metrics.auc(recall, precision)
            self.pr_area_list.append(pr_area)

            self.predictions_val.append(predictions)
            self.truths_val.append(truths)

            if pr_area >= self.best_pr_area:
                self.best_pr_area = pr_area
                self.best_epoch = self.last_epoch
                self.epochs_since_best = 0
                self.save(path, True)
            else:
                self.epochs_since_best += 1
                self.save(path, False)

        # Save test results
        else:
            self.precision_test = precision
            self.recall_test = recall
            self.f1_measure_test = f1_measure
            pr_area = metrics.auc(recall, precision)
            self.pr_area_test = pr_area
            self.predictions_test = predictions
            self.truths_test = truths
            self.save(path, False)

        # print "Precision: ", precision
        # print "Recall: ", recall
        # print "F1: ", f1_measure
        print "AUC: ", pr_area

    def save(self, path, save_weights):
        """
        Saves the weights and stats.
        Weights are only saved if save_weights is True (i.e. only if the pr_area is the best so far)
        """
        print('Saving Model...')

        # Weights
        if save_weights:
            np.savez(path + '_weights.npz', *lasagne.layers.get_all_param_values(self.l_out))

        # Stats
        stats = ModelStats(self.last_epoch, self.best_epoch, self.best_pr_area, self.epochs_since_best,
                           self.precision_list, self.recall_list, self.f1_measure_list, self.pr_area_list,
                           self.learning_rate, self.sentence_encoder, self.units_sentence_encoder,
                           self.cnn_filter_size, self.word_emb_size, self.pos_emb_size, self.dropout_prob,
                           self.precision_test, self.recall_test, self.f1_measure_test, self.pr_area_test,
                           self.predictions_val, self.predictions_test, self.truths_val, self.truths_test)
        f = open(path + '_stats.pk', 'wb')
        pickle.dump(stats, f, 2)

    def load(self, path):
        """"Load existing weights as a starting point."""

        print('Loading existing weights...')

        # Weights
        with np.load(path + '_weights.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.l_out, param_values)

        # Stats
        if os.path.exists(path + '_stats.pk'):
            print('Loading existing stats...')
            stats = pickle.load(open(path + '_stats.pk', 'rb'))
            self.last_epoch = stats.last_epoch
            self.best_epoch = stats.best_epoch
            self.best_pr_area = stats.best_pr_area
            self.epochs_since_best = stats.epochs_since_best
            self.precision_list = stats.precision_list
            self.recall_list = stats.recall_list
            self.f1_measure_list = stats.f1_measure_list
            self.pr_area_list = stats.pr_area_list
            self.precision_test = stats.precision_test
            self.recall_test = stats.recall_test
            self.f1_measure_test = stats.f1_measure_test
            self.pr_area_test = stats.pr_area_test
            self.learning_rate = stats.learning_rate
            self.sentence_encoder = stats.sentence_encoder
            self.units_sentence_encoder = stats.units_sentence_encoder
            self.cnn_filter_size = stats.cnn_filter_size
            self.word_emb_size = stats.word_emb_size
            self.pos_emb_size = stats.pos_emb_size
            self.dropout_prob = stats.dropout_prob
            self.predictions_val = stats.predictions_val
            self.predictions_test = stats.predictions_test
            self.truths_val = stats.truths_val
            self.truths_test = stats.truths_test
        else:
            print('NOTE: no existing stats found!')
