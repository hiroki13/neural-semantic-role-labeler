import gzip
import cPickle

import numpy as np
import theano

from ..ling.vocab import Vocab, PAD, UNK, UNDER_BAR, VERB, BE, SLASH


def load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


def load_conll(path, data_size=1000000, file_encoding='utf-8'):
    if path is None:
        return None

    corpus = []
    with open(path) as f:
        sent = []
        for line in f:
            es = line.rstrip().split()
            if len(es) > 1:
                word = es[0].decode(file_encoding).lower()
                tag = es[1].decode(file_encoding)
                syn = es[2].decode(file_encoding)
                ne = es[3].decode(file_encoding)
                prd = es[4].decode(file_encoding)
                prop = []

                if len(es) > 5:
                    prop = es[5:]
                sent.append((word, tag, syn, ne, prd, prop))
            else:
                corpus.append(sent)
                sent = []

            if len(corpus) >= data_size:
                break

        if sent:
            corpus.append(sent)

    return corpus


def load_pos_tagged_corpus(path, data_size=1000000, file_encoding='utf-8'):
    """
    An example sent: A_DT tropical_JJ storm_NN rapidly_RB ...
    """
    if path is None:
        return None

    corpus = []
    with open(path) as f:
        for line in f:
            sent = []
            for token in line.rstrip().split():
                token = ''.join([e.decode(file_encoding) for e in token.split()])
                es = token.split(UNDER_BAR)

                if len(es) != 2:
                    es = [UNDER_BAR, es[-1]]

                word = es[0].lower()
                tag = es[1].decode(file_encoding)
                syn = ''
                ne = ''
                prd = VERB if tag.startswith(VERB) and word not in BE else SLASH
                prop = []
                sent.append((word, tag, syn, ne, prd, prop))
            corpus.append(sent)

            if len(corpus) >= data_size:
                break

    return corpus


def load_init_emb(init_emb):
    vocab = Vocab()
    vocab.add_word(PAD)
    vocab.add_word(UNK)

    vec_dict = {}
    with open(init_emb) as emb_file:
        for line in emb_file:
            line = line.strip().decode('utf-8').split()
            word = line[0]

            if word[1:-1] == UNK:
                word = UNK

            vocab.add_word(word)
            vec_dict[vocab.get_id(word)] = line[1:]

    emb = _create_init_emb(vocab, vec_dict)

    if len(emb[vocab.get_id(PAD)]) == 0:
        emb[vocab.get_id(PAD)] = np.zeros(len(vec_dict.values()[0]), dtype=theano.config.floatX)

    if len(emb[vocab.get_id(UNK)]) == 0:
        emb[vocab.get_id(UNK)] = _average_vector(emb)

    emb = np.asarray(emb, dtype=theano.config.floatX)
    assert emb.shape[0] == vocab.size()
    return emb, vocab


def _create_init_emb(vocab, vec):
    emb = [[] for i in xrange(vocab.size())]
    for k, v in vec.items():
        emb[k] = v
    return emb


def _average_vector(emb):
    return np.mean(np.asarray(emb[2:], dtype=theano.config.floatX), axis=0)
