__author__ = 'hiroki'

import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, relu, build_shared_zeros, sample_weights
from crf import CRFLayer


def layers(x, batch, n_fin, n_h, n_y, n_layers=1):
    params = []

    for i in xrange(n_layers):
        if i == 0:
            layer = FirstLayer(n_i=n_fin, n_h=n_h)
            layer_input = relu(T.dot(x.dimshuffle(1, 0, 2), layer.W))
            h0 = layer.h0 * T.ones((batch, n_h))  # h0: 1D: Batch, 2D: n_h
        else:
            layer = Layer(n_i=n_h*2, n_h=n_h)
            layer_input = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))[::-1]  # h: 1D: n_words, 2D: Batch, 3D n_h
            h0 = layer_input[0]

        xr = T.dot(layer_input, layer.W_xr)
        xz = T.dot(layer_input, layer.W_xz)
        xh = T.dot(layer_input, layer.W_xh)

        h, _ = theano.scan(fn=layer.forward, sequences=[xr, xz, xh], outputs_info=[h0])
        params.extend(layer.params)

    layer = CRFLayer(n_i=n_h*2, n_h=n_y)
    params.extend(layer.params)
    h = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))

    if n_layers % 2 == 0:
        emit = h[::-1]
    else:
        emit = h

    return params, layer, emit


class Layer(object):
    def __init__(self, n_i=32, n_h=32, activation=tanh):
        self.activation = activation

        self.W = theano.shared(sample_weights(n_i, n_h))

        self.W_xr = theano.shared(sample_weights(n_h, n_h))
        self.W_hr = theano.shared(sample_weights(n_h, n_h))

        self.W_xz = theano.shared(sample_weights(n_h, n_h))
        self.W_hz = theano.shared(sample_weights(n_h, n_h))

        self.W_xh = theano.shared(sample_weights(n_h, n_h))
        self.W_hh = theano.shared(sample_weights(n_h, n_h))

        self.params = [self.W, self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]

    def forward(self, xr_t, xz_t, xh_t, h_tm1):
        r_t = sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(xz_t + T.dot(h_tm1, self.W_hz))
        h_hat_t = self.activation(xh_t + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t


class FirstLayer(object):
    def __init__(self, n_i=32, n_h=32, activation=tanh):
        self.activation = activation
        self.h0 = build_shared_zeros(n_h)

        self.W = theano.shared(sample_weights(n_i, n_h))

        self.W_xr = theano.shared(sample_weights(n_h, n_h))
        self.W_hr = theano.shared(sample_weights(n_h, n_h))

        self.W_xz = theano.shared(sample_weights(n_h, n_h))
        self.W_hz = theano.shared(sample_weights(n_h, n_h))

        self.W_xh = theano.shared(sample_weights(n_h, n_h))
        self.W_hh = theano.shared(sample_weights(n_h, n_h))

        self.params = [self.W, self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]

    def forward(self, xr_t, xz_t, xh_t, h_tm1):
        r_t = sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(xz_t + T.dot(h_tm1, self.W_hz))
        h_hat_t = self.activation(xh_t + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t
