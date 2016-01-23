__author__ = 'hiroki'

import theano
import theano.tensor as T

import lstm
import gru
from nn_utils import L2_sqr
from crf import y_prob, vitabi
from optimizers import adam


class RNN(object):
    def __init__(self, unit, x, d, n_layers, n_in, n_h, n_y, reg=0.0001):

        if unit == 'lstm':
            self.layers = lstm.layers
        else:
            self.layers = gru.layers

        self.x = x  # x: 1D: batch_size, 2D: n_words, 3D: n_fin
        self.d = d  # d: 1D: batch_size, 2D: n_words

        n_fin = n_in * 7 + 1
        batch = T.cast(self.d.shape[0], dtype='int32')

        params, o_layer, emit = self.layers(x=self.x, batch=batch, n_fin=n_fin, n_h=n_h, n_y=n_y, n_layers=n_layers)

        self.p_y = y_prob(o_layer, emit, self.d.dimshuffle(1, 0), batch)
        self.y_pred = vitabi(o_layer, emit, batch)

        self.nll = - T.mean(self.p_y)
        self.L2_sqr = L2_sqr(params)
        self.cost = self.nll + reg * self.L2_sqr / 2.
        self.errors = T.neq(self.y_pred, self.d)

        self.g = T.grad(self.cost, params)
        self.updates = adam(params, self.g)
