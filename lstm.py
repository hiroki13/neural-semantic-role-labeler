__author__ = 'hiroki'

import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, relu, build_shared_zeros, sample_weights
from crf import CRFLayer


def layers(x, batch, n_fin, n_h, n_y, n_layers=1):
    params = []

    for i in xrange(n_layers):
        if i == 0:
            layer = LSTMFirstLayer(n_i=n_fin, n_h=n_h)
            layer_input = relu(T.dot(x.dimshuffle(1, 0, 2), layer.W))  # x: 1D: Batch, 2D: n_words, 3D: n_fin
            h0 = layer.h0 * T.ones((batch, n_h))  # h0: 1D: Batch, 2D: n_h
            c0 = layer.c0 * T.ones((batch, n_h))  # c0: 1D: Batch, 2D: n_h
        else:
            layer = LSTMLayer(n_i=n_h*2, n_h=n_h)
            layer_input = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))[::-1]  # h: 1D: n_words, 2D: Batch, 3D n_h
            h0 = layer_input[0]
            c0 = c[-1]

        xi = T.dot(layer_input, layer.W_xi)  # layer_input: 1D: n_words, 2D: Batch, 3D: n_fin
        xf = T.dot(layer_input, layer.W_xf)
        xc = T.dot(layer_input, layer.W_xc)
        xo = T.dot(layer_input, layer.W_xo)

        [h, c], _ = theano.scan(fn=layer.forward,
                                sequences=[xi, xf, xc, xo],
                                outputs_info=[h0, c0])

        params.extend(layer.params)

    layer = CRFLayer(n_i=n_h*2, n_h=n_y)
    params.extend(layer.params)
    h = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))

    if n_layers % 2 == 0:
        emit = h[::-1]
    else:
        emit = h

    return params, layer, emit


class LSTMLayer(object):
    def __init__(self, n_i, n_h, activation=tanh):
        self.activation = activation
        self.W = theano.shared(sample_weights(n_i, n_h))

        """input gate parameters"""
        self.W_xi = theano.shared(sample_weights(n_h, n_h))
        self.W_hi = theano.shared(sample_weights(n_h, n_h))
        self.W_ci = theano.shared(sample_weights(n_h))

        """forget gate parameters"""
        self.W_xf = theano.shared(sample_weights(n_h, n_h))
        self.W_hf = theano.shared(sample_weights(n_h, n_h))
        self.W_cf = theano.shared(sample_weights(n_h))

        """cell parameters"""
        self.W_xc = theano.shared(sample_weights(n_h, n_h))
        self.W_hc = theano.shared(sample_weights(n_h, n_h))

        """output gate parameters"""
        self.W_xo = theano.shared(sample_weights(n_h, n_h))
        self.W_ho = theano.shared(sample_weights(n_h, n_h))
        self.W_co = theano.shared(sample_weights(n_h))

        self.params = [self.W, self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf,
                       self.W_xc, self.W_hc, self.W_xo, self.W_ho, self.W_co]

    def forward(self, xi_t, xf_t, xc_t, xo_t, h_tm1, c_tm1):
        i_t = sigmoid(xi_t + T.dot(h_tm1, self.W_hi) + c_tm1 * self.W_ci)
        f_t = sigmoid(xf_t + T.dot(h_tm1, self.W_hf) + c_tm1 * self.W_cf)
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, self.W_hc))
        o_t = sigmoid(xo_t + T.dot(h_tm1, self.W_ho) + c_t * self.W_co)
        h_t = o_t * self.activation(c_t)
        return h_t, c_t


class LSTMFirstLayer(object):
    def __init__(self, n_i, n_h, activation=tanh):
        self.activation = activation
        self.c0 = build_shared_zeros(n_h)
        self.h0 = self.activation(self.c0)

        self.W = theano.shared(sample_weights(n_i, n_h))

        """input gate parameters"""
        self.W_xi = theano.shared(sample_weights(n_h, n_h))
        self.W_hi = theano.shared(sample_weights(n_h, n_h))
        self.W_ci = theano.shared(sample_weights(n_h))

        """forget gate parameters"""
        self.W_xf = theano.shared(sample_weights(n_h, n_h))
        self.W_hf = theano.shared(sample_weights(n_h, n_h))
        self.W_cf = theano.shared(sample_weights(n_h))

        """cell parameters"""
        self.W_xc = theano.shared(sample_weights(n_h, n_h))
        self.W_hc = theano.shared(sample_weights(n_h, n_h))

        """output gate parameters"""
        self.W_xo = theano.shared(sample_weights(n_h, n_h))
        self.W_ho = theano.shared(sample_weights(n_h, n_h))
        self.W_co = theano.shared(sample_weights(n_h))

        self.params = [self.W, self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf,
                       self.W_xc, self.W_hc, self.W_xo, self.W_ho, self.W_co]

    def forward(self, xi_t, xf_t, xc_t, xo_t, h_tm1, c_tm1):
        """
        :param x_t: 1D: Batch, 2D: n_in
        :param h_tm1: 1D: Batch, 2D: n_h
        :param c_tm1: 1D: Batch, 2D; n_h
        :return: h_t: 1D: Batch, 2D: n_h
        :return: c_t: 1D: Batch, 2D: n_h
        """
        i_t = sigmoid(xi_t + T.dot(h_tm1, self.W_hi) + c_tm1 * self.W_ci)
        f_t = sigmoid(xf_t + T.dot(h_tm1, self.W_hf) + c_tm1 * self.W_cf)
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, self.W_hc))
        o_t = sigmoid(xo_t + T.dot(h_tm1, self.W_ho) + c_t * self.W_co)
        h_t = o_t * self.activation(c_t)
        return h_t, c_t
