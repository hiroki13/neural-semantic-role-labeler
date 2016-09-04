import theano
import theano.tensor as T

from crf import CRF
from nn_utils import sigmoid, tanh, relu, build_shared_zeros, sample_weights


def gru_layers(x, batch, n_fin, n_h, n_y, n_layers=1):
    params = []

    for i in xrange(n_layers):
        if i == 0:
            layer = GRU(n_i=n_fin, n_h=n_h)
            layer_input = relu(T.dot(x.dimshuffle(1, 0, 2), layer.W))
            # h0: 1D: Batch, 2D: n_h
            h0 = T.zeros((batch, n_h), dtype=theano.config.floatX)
        else:
            layer = GRU(n_i=n_h * 2, n_h=n_h)
            # h: 1D: n_words, 2D: Batch, 3D n_h
            layer_input = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))[::-1]
            h0 = layer_input[0]

        xr = T.dot(layer_input, layer.W_xr)
        xz = T.dot(layer_input, layer.W_xz)
        xh = T.dot(layer_input, layer.W_xh)

        h, _ = theano.scan(fn=layer.forward, sequences=[xr, xz, xh], outputs_info=[h0])
        params.extend(layer.params)

    layer = CRF(n_i=n_h * 2, n_h=n_y)
    params.extend(layer.params)
    h = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))

    if n_layers % 2 == 0:
        emit = h[::-1]
    else:
        emit = h

    return params, layer, emit


class GRU(object):

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

    def forward_all(self, x, h0):
        xr = T.dot(x, self.W_xr)
        xz = T.dot(x, self.W_xz)
        xh = T.dot(x, self.W_xh)
        h, _ = theano.scan(fn=self.forward, sequences=[xr, xz, xh], outputs_info=[h0])
        return h

    def dot(self, x):
        return relu(T.dot(x, self.W))


def lstm_layers(x, batch, n_fin, n_h, n_y, n_layers=1):
    params = []

    for i in xrange(n_layers):
        if i == 0:
            layer = LSTM(n_i=n_fin, n_h=n_h)
            layer_input = relu(T.dot(x.dimshuffle(1, 0, 2), layer.W))  # x: 1D: Batch, 2D: n_words, 3D: n_fin
            h0 = layer.h0 * T.ones((batch, n_h))  # h0: 1D: Batch, 2D: n_h
            c0 = layer.c0 * T.ones((batch, n_h))  # c0: 1D: Batch, 2D: n_h
        else:
            layer = LSTM(n_i=n_h * 2, n_h=n_h)
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

    layer = CRF(n_i=n_h * 2, n_h=n_y)
    params.extend(layer.params)
    h = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))

    if n_layers % 2 == 0:
        emit = h[::-1]
    else:
        emit = h

    return params, layer, emit


class LSTM(object):
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
