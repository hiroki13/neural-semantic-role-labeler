__author__ = 'hiroki'

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, relu, build_shared_zeros


class LSTM(object):
    def __init__(self, x, d, da, n_layers, op, n_in=32, n_h=32, n_ya=45, n_y=105, lr1=0.01, lr2=0.001, L2_reg=0.0001,
                 activation=tanh):

        self.x = x  # x: 1D: batch_size, 2D: n_words, 3D: n_fin
        self.d = d.dimshuffle(1, 0)  # d: 1D: batch_size, 2D: n_words

        self.n_layers = n_layers
        self.n_in = n_in
        self.n_h = n_h
        self.n_ya = n_ya
        self.n_y = n_y
        self.n_words = self.x.shape[1]
        self.n_fin = self.n_in * 7 + 1
        self.lr1 = lr1
        self.lr2 = lr2
        self.activation = activation
        self.batch = T.cast(self.d.shape[1], dtype='int32')

        """ BIO layers and parameters """
        self.params_bio, self.bio_log_p, self.bio_x, self.bio_c = self.layers_bio(x=self.x, n_layers=1)

        """ SRL layers and parameters """
        self.last_layer, self.params_srl, self.y, self.emit = self.layers(x=self.bio_x.dimshuffle(1, 0, 2), h=self.bio_log_p, c=self.bio_c, n_layers=n_layers)

        """ Regularization """
        self.L2_sqr1 = 0.0
        for p in self.params_bio:
            self.L2_sqr1 += (p ** 2).sum()

        self.L2_sqr2 = 0.0
        for p in self.params_srl:
            self.L2_sqr2 += (p ** 2).sum()

        """ BIO Computation """
        output_span = self.bio_log_p.dimshuffle(1, 0, 2)
        self.ya = output_span.reshape((output_span.shape[0]*output_span.shape[1], -1))[T.arange(da.shape[0]*da.shape[1]), da.flatten()]
        self.ya_pred = T.argmax(output_span, axis=2)
        self.nll1 = - T.mean(self.ya)
        self.errors_a = T.neq(self.ya_pred, da)

        """ SRL Computation """
        self.y_pred = self.vitabi(self.last_layer, self.emit)
        self.nll2 = - T.mean(self.y)
        self.errors = T.neq(self.y_pred, d)

        """ Objective Function """
        self.nll = self.nll1 + self.nll2
        self.cost1 = self.nll + L2_reg * (self.L2_sqr1+self.L2_sqr2) / 2
        self.cost2 = self.nll2 + L2_reg * self.L2_sqr2 / 2
        self.cost3 = self.nll1 + L2_reg * self.L2_sqr1 / 2

        self.g0 = T.grad(self.cost1, self.params_bio)
        self.g1 = T.grad(self.cost1, self.params_srl)
        self.g2 = T.grad(self.cost3, self.params_bio)

        self.updates1 = self.adam(self.g2)
        self.updates2 = self.adam(self.g0, self.g1)

    def layers_bio(self, x, n_layers=1):
        layers = []
        params = []
        layer_input = []
        layer_output = []
        layer_last_c = []

        for i in xrange(n_layers):
            if i == 0:
                layer = FirstLayer(n_i=self.n_fin, n_h=self.n_h)
                layer_input.append(relu(T.dot(x, layer.W)))  # x: 1D: Batch, 2D: n_words, 3D: n_fin
                h0 = T.repeat(layer.h0.dimshuffle('x', 0), self.batch, 0)  # h0: 1D: Batch, 2D: n_h
                c0 = T.repeat(layer.c0.dimshuffle('x', 0), self.batch, 0)  # c0: 1D: Batch, 2D: n_h
                params.extend(layer.params)
            else:
                layer = Layer(n_i=self.n_h, n_h=self.n_h)
                x_n = relu(T.dot(T.concatenate([layer_input[-1].dimshuffle(1, 0, 2), layer_output[-1]], 2), layer.W))[::-1]  # layer_output[-1]: 1D: n_words, 2D: Batch, 3D n_h
                layer_input.append(x_n.dimshuffle(1, 0, 2))
                h0 = x_n[0]
                c0 = layer_last_c[-1][-1]
                params.extend(layer.params)

            xi = T.dot(layer_input[-1], layer.W_xi).dimshuffle(1, 0, 2)  # layer_input: 1D: Batch, 2D: n_words, 3D: n_fin
            xf = T.dot(layer_input[-1], layer.W_xf).dimshuffle(1, 0, 2)
            xc = T.dot(layer_input[-1], layer.W_xc).dimshuffle(1, 0, 2)
            xo = T.dot(layer_input[-1], layer.W_xo).dimshuffle(1, 0, 2)

            [h, c], _ = theano.scan(fn=layer.forward,
                                    sequences=[xi, xf, xc, xo],
                                    outputs_info=[h0, c0])

            layers.append(layer)
            layer_output.append(h)
            layer_last_c.append(c)

        layer = BIOLayer(n_i=self.n_h, n_h=self.n_ya)
        h1 = T.concatenate([layer_input[-1].dimshuffle(1, 0, 2), layer_output[-1]], 2)  # h1: 1D: n_words, 2D: Batch, 3D: n_h*2
        hidden = T.dot(h1, layer.W)  # hidden: 1D: n_words, 2D: Batch, 3D: n_ya

        z = logsumexp(hidden, axis=2)
        emit = (hidden - z)

        if n_layers % 2 == 0:
            emit = emit[::-1]

        layers.append(layer)
        params.extend(layer.params)
        layer_output.append(emit)

        return params, emit, h1, layer_last_c[-1]

    def layers(self, x, h, c, n_layers=1):
        params = []

        if n_layers > 1:
            n_i = self.n_h * 2
            for i in xrange(n_layers-1):
                if i == 0:
                    layer = Layer(n_i=self.n_h*2+self.n_ya, n_h=self.n_h)
                else:
                    layer = Layer(n_i=self.n_h*2, n_h=self.n_h)

                params.extend(layer.params)
                x = relu(T.dot(T.concatenate([x.dimshuffle(1, 0, 2), h], 2), layer.W))[::-1].dimshuffle(1, 0, 2)  # x: 1D: n_words, 2D: batch, 3D: n_h
                h0 = x.dimshuffle(1, 0, 2)[0]
                xi = T.dot(x, layer.W_xi).dimshuffle(1, 0, 2)  # layer_input: 1D: Batch, 2D: n_words, 3D: n_fin
                xf = T.dot(x, layer.W_xf).dimshuffle(1, 0, 2)
                xc = T.dot(x, layer.W_xc).dimshuffle(1, 0, 2)
                xo = T.dot(x, layer.W_xo).dimshuffle(1, 0, 2)
                [h, c], _ = theano.scan(fn=layer.forward, sequences=[xi, xf, xc, xo], outputs_info=[h0, c[-1]])
        else:
            n_i = self.n_h * 2 + self.n_ya

        layer = LastLayer(n_i=n_i, n_y=self.n_y)
        params.extend(layer.params)
        x = T.dot(T.concatenate([x.dimshuffle(1, 0, 2), h], 2), layer.W)  # x: 1D: n_words, 2D: Batch, 3D: n_y

        if n_layers % 2 == 0:
            emit = x[::-1]
        else:
            emit = x

        log_p_d = self.forward(layer, emit)

        return layer, params, log_p_d, emit

    def forward(self, layer, emit):
        """
        :param emit: 1D: n_words, 2D: Batch, 3D: n_y
        :return:
        """

        def forward_step(e_t, d_t, d_prev, d_score_prev, z_scores_prev, trans):
            """
            :param e_t: 1D: Batch, 2D: n_y
            :param d_t: 1D: Batch
            :param d_prev: 1D: Batch
            :param d_score_prev: 1D: Batch
            :param z_scores_prev: 1D: Batch, 2D: n_y
            :param trans: 1D: n_y, 2D, n_y
            """
            d_score_t = d_score_prev + trans[d_t, d_prev] + e_t[T.arange(self.batch), d_t]  # 1D: Batch
            z_sum = T.repeat(z_scores_prev, T.cast(trans.shape[0], dtype='int32'), 0).reshape(
                (self.batch, trans.shape[0], trans.shape[1])) + trans  # 1D: Batch, 2D: n_y, 3D: n_y
            z_scores_t = logsumexp(z_sum, axis=2).reshape((self.batch, e_t.shape[1])) + e_t  # 1D: Batch, 2D: n_y
            return d_t, d_score_t, z_scores_t

        d_score0 = layer.BOS[self.d[0]] + emit[0][T.arange(self.batch), self.d[0]]  # 1D: Batch
        z_scores0 = layer.BOS + emit[0]  # 1D: Batch, 2D: n_y

        [_, d_scores, z_scores], _ = theano.scan(fn=forward_step,
                                                 sequences=[emit[1:], self.d[1:]],
                                                 outputs_info=[self.d[0], d_score0, z_scores0],
                                                 non_sequences=layer.W_t)

        d_score = d_scores[-1]
        z_score = logsumexp(z_scores[-1], axis=1).flatten()

        return d_score - z_score

    def vitabi(self, layer, emit):
        def forward(e_t, score_prev, trans):
            """
            :param e_t: 1D: Batch, 2D: n_y
            :param scores_prev: 1D: Batch, 2D: n_y
            :param trans: 1D: n_y, 2D, n_y
            """
            score = T.repeat(score_prev, T.cast(trans.shape[0], dtype='int32'), 0).reshape(
                (self.batch, trans.shape[0], trans.shape[1])) + trans + e_t.dimshuffle(0, 1, 'x')
            max_scores_t = T.max(score, axis=2)
            max_nodes_t = T.cast(T.argmax(score, axis=2), dtype='int32')
            return max_scores_t, max_nodes_t

        def backward(nodes_t, max_node_t):
            return nodes_t[T.arange(self.batch), max_node_t]

        scores0 = layer.BOS + emit[0]
        [max_scores, max_nodes], u5 = theano.scan(fn=forward,
                                                  sequences=[emit[1:]],
                                                  outputs_info=[scores0, None],
                                                  non_sequences=layer.W_t)
        max_node = T.cast(T.argmax(max_scores[-1], axis=1), dtype='int32')

        nodes, _ = theano.scan(fn=backward,
                               sequences=max_nodes[::-1],
                               outputs_info=max_node)

        return T.concatenate([nodes[::-1].dimshuffle(1, 0), max_node.dimshuffle((0, 'x'))], 1)

    def adam(self, grads0, grads1=None, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
        updates = OrderedDict()
        i = theano.shared(np.float32(0))
        i_t = i + 1.

        for p, g in zip(self.params_bio, grads0):
#            g = grad_clipping(g, 50.)
            v = build_shared_zeros(p.get_value(True).shape)
            r = build_shared_zeros(p.get_value(True).shape)

            v_t = (b1 * v) + (1. - b1) * g
            r_t = (b2 * r) + (1. - b2) * T.sqr(g)

            r_hat = lr / (T.sqrt(r_t / (1 - b2 ** i_t)) + e)
            v_hat = v / (1 - b1 ** i_t)

            p_t = p - r_hat * v_hat
            updates[v] = v_t
            updates[r] = r_t
            updates[p] = p_t

        if grads1 is not None:
            for p, g in zip(self.params_srl, grads1):
#                g = grad_clipping(g, 50.)
                v = build_shared_zeros(p.get_value(True).shape)
                r = build_shared_zeros(p.get_value(True).shape)

                v_t = (b1 * v) + (1. - b1) * g
                r_t = (b2 * r) + (1. - b2) * T.sqr(g)

                r_hat = lr / (T.sqrt(r_t / (1 - b2 ** i_t)) + e)
                v_hat = v / (1 - b1 ** i_t)

                p_t = p - r_hat * v_hat
                updates[v] = v_t
                updates[r] = r_t
                updates[p] = p_t

        updates[i] = i_t
        return updates


def grad_clipping(g, s):
    g_norm = T.abs_(g)
    return T.switch(g_norm > s, (s * g) / g_norm, g)


def sample_weights(size_x, size_y=0):
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
#    if size_y == 0:
#        W = np.asarray(np.random.uniform(low=-0.08,
#                                         high=0.08,
#                                         size=size_x),
#                       dtype=theano.config.floatX)
#    else:
#        W = np.asarray(np.random.uniform(low=-0.08,
#                                         high=0.08,
#                                         size=(size_x, size_y)),
#                       dtype=theano.config.floatX)
    return W


def logsumexp(x, axis):
    """
    :param x: 1D: batch, 2D: n_y, 3D: n_y
    :return:
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


class Layer(object):
    def __init__(self, n_i=32, n_h=32, activation=tanh):
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


class FirstLayer(object):
    def __init__(self, n_i=32, n_h=32, activation=tanh):
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

        self.params = [self.c0, self.W, self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf,
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


class BIOLayer(object):
    def __init__(self, n_i, n_h):
        self.W = theano.shared(sample_weights(n_i*2, n_h))
        self.params = [self.W]


class LastLayer(object):
    def __init__(self, n_i, n_y):
        self.W = theano.shared(sample_weights(n_i, n_y))
        self.W_t = theano.shared(sample_weights(n_y, n_y))
        self.BOS = theano.shared(sample_weights(n_y))
        self.params = [self.W, self.W_t, self.BOS]
