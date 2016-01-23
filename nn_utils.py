__author__ = 'hiroki'

import numpy as np
import theano
import theano.tensor as T


def relu(x):
    return T.maximum(0, x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def build_shared_zeros(shape):
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        borrow=True
    )


def sample_weights_bengio(size_x, size_y=0):
    if size_y == 0:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=size_x),
                       dtype=theano.config.floatX)
    else:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=(size_x, size_y)),
                       dtype=theano.config.floatX)
    return W


def sample_weights(size_x, size_y=0):
    if size_y == 0:
        W = np.asarray(np.random.uniform(low=-0.08, high=0.08, size=size_x),
                       dtype=theano.config.floatX)
    else:
        W = np.asarray(np.random.uniform(low=-0.08, high=0.08, size=(size_x, size_y)),
                       dtype=theano.config.floatX)
    return W


def logsumexp(x, axis):
    """
    :param x: 1D: batch, 2D: n_y, 3D: n_y
    :return:
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def L2_sqr(params):
    sqr = 0.0
    for p in params:
        sqr += (p ** 2).sum()
    return sqr
