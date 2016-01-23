__author__ = 'hiroki'


import theano
import theano.tensor as T

from nn_utils import logsumexp, sample_weights


class CRFLayer(object):
    def __init__(self, n_i, n_h):
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.W_t = theano.shared(sample_weights(n_h, n_h))
        self.BOS = theano.shared(sample_weights(n_h))
        self.params = [self.W, self.W_t, self.BOS]


def y_prob(layer, emit, d, batch):
    """
    :param emit: 1D: n_words, 2D: Batch, 3D: n_y
    :return: gradient of cross entropy: 1D: Batch
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
        d_score_t = d_score_prev + trans[d_t, d_prev] + e_t[T.arange(batch), d_t]  # 1D: Batch
        z_sum = z_scores_prev.dimshuffle(0,'x',1) + trans  # 1D: Batch, 2D: n_y, 3D: n_y
        z_scores_t = logsumexp(z_sum, axis=2).reshape(e_t.shape) + e_t  # 1D: Batch, 2D: n_y
        return d_t, d_score_t, z_scores_t

    d_score0 = layer.BOS[d[0]] + emit[0][T.arange(batch), d[0]]  # 1D: Batch
    z_scores0 = layer.BOS + emit[0]  # 1D: Batch, 2D: n_y

    [_, d_scores, z_scores], _ = theano.scan(fn=forward_step,
                                             sequences=[emit[1:], d[1:]],
                                             outputs_info=[d[0], d_score0, z_scores0],
                                             non_sequences=layer.W_t)

    d_score = d_scores[-1]
    z_score = logsumexp(z_scores[-1], axis=1).flatten()

    return d_score - z_score


def vitabi(layer, emit, batch):
    def forward(e_t, score_prev, trans):
        """
        :param e_t: 1D: Batch, 2D: n_y
        :param scores_prev: 1D: Batch, 2D: n_y
        :param trans: 1D: n_y, 2D, n_y
        """
        score = score_prev.dimshuffle(0, 'x', 1) + trans + e_t.dimshuffle(0, 1, 'x')
        max_scores_t, max_nodes_t = T.max_and_argmax(score, axis=2)
        return max_scores_t, T.cast(max_nodes_t, dtype='int32')

    def backward(nodes_t, max_node_t):
        return nodes_t[T.arange(batch), max_node_t]

    scores0 = layer.BOS + emit[0]
    [max_scores, max_nodes], _ = theano.scan(fn=forward,
                                             sequences=[emit[1:]],
                                             outputs_info=[scores0, None],
                                             non_sequences=layer.W_t)
    max_last_node = T.cast(T.argmax(max_scores[-1], axis=1), dtype='int32')

    nodes, _ = theano.scan(fn=backward,
                           sequences=max_nodes[::-1],
                           outputs_info=max_last_node)

    return T.concatenate([nodes[::-1].dimshuffle(1, 0), max_last_node.dimshuffle((0, 'x'))], 1)

