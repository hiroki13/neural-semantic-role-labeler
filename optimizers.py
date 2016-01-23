__author__ = 'hiroki'

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from nn_utils import build_shared_zeros


def grad_clipping(g, t):
    return T.switch(T.abs_(g) >= t, t / T.abs_(g) * g, g)


def sgd(cost, params, emb=None, sub_emb=None, lr=0.1):
    updates = OrderedDict()

    """update sub-tensor of embeddings"""
    if emb:
        g = T.grad(cost, sub_emb)
        updates[emb] = T.inc_subtensor(sub_emb, -lr * g)

    """update parameters"""
    grads0 = T.grad(cost, params[0])
    for p, g in zip(params[0], grads0):
        updates[p] = p - lr * g

    """update parameters"""
    grads1 = T.grad(cost, params[1])
    for p, g in zip(params[1], grads1):
        updates[p] = p - lr * g

    return updates


def ada_grad(cost, params, emb=None, sub_emb=None, w=None, lr=0.1, eps=1.):
    updates = OrderedDict()

    """update sub-tensor of embeddings"""
    if emb:
        p = emb
        g = T.grad(cost, sub_emb)
        r = build_shared_zeros(p.get_value(True).shape)
        r_sub = r[w]
        r_sub_t = r_sub + T.sqr(g)
        r_t = T.set_subtensor(r_sub, r_sub_t)
        p_t = T.inc_subtensor(sub_emb, - (lr / (T.sqrt(r_sub_t) + eps)) * g)
        updates[r] = r_t
        updates[p] = p_t

    """update parameters"""
    grads0 = T.grad(cost, params[0])
    for p, g in zip(params[0], grads0):
        r = build_shared_zeros(p.get_value(True).shape)
        r_t = r + T.sqr(g)
        p_t = p - (lr / (T.sqrt(r_t) + eps)) * g
        updates[r] = r_t
        updates[p] = p_t

    """update parameters"""
    grads1 = T.grad(cost, params[1])
    for p, g in zip(params[1], grads1):
        r = build_shared_zeros(p.get_value(True).shape)
        r_t = r + T.sqr(g)
        p_t = p - (lr / (T.sqrt(r_t) + eps)) * g
        updates[r] = r_t
        updates[p] = p_t
    return updates


def ada_delta(cost, params, emb, x, w, b=0.999, eps=1e-8):
    updates = OrderedDict()
    grads = T.grad(cost, params)

    """update sub-tensor of embeddings"""
    p = emb
    g = T.grad(cost, x)

    r = build_shared_zeros(p.get_value(True).shape)
    v = build_shared_zeros(p.get_value(True).shape)
    s = build_shared_zeros(p.get_value(True).shape)
    r_sub = r[w]
    v_sub = v[w]
    s_sub = s[w]

    r_sub_t = b * r_sub + (1 - b) * T.sqr(g)
    v_sub_t = (T.sqrt(s_sub) + eps) / (T.sqrt(r_sub) + eps) * g
    s_sub_t = b * s_sub + (1 - b) * T.sqr(v_sub_t)
    updates[r] = T.set_subtensor(r_sub, r_sub_t)
    updates[v] = T.set_subtensor(v_sub, v_sub_t)
    updates[s] = T.set_subtensor(s_sub, s_sub_t)
    updates[p] = T.inc_subtensor(x, -v_sub_t)

    """update parameters"""
    for p, g in zip(params, grads):
        r = build_shared_zeros(p.get_value(True).shape)
        v = build_shared_zeros(p.get_value(True).shape)
        s = build_shared_zeros(p.get_value(True).shape)
        r_t = b * r + (1 - b) * T.sqr(g)
        v_t = (T.sqrt(s) + eps) / (T.sqrt(r) + eps) * g
        s_t = b * s + (1 - b) * T.sqr(v_t)
        p_t = p - v_t
        updates[r] = r_t
        updates[v] = v_t
        updates[s] = s_t
        updates[p] = p_t
    return updates


def adam(params, grads, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    updates = OrderedDict()
    i = theano.shared(np.float32(0))
    i_t = i + 1.

    for p, g in zip(params, grads):
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
