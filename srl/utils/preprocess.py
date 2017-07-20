from copy import deepcopy
from collections import Counter

import numpy as np
import theano

from io_utils import say
from ..ling.vocab import Vocab, UNK, VERB, SLASH, UNDER_BAR


def convert_to_conll(text, file_encoding='utf-8'):
    # text: [u'I_PRP am_VBP John_NNP ._.']
    corpus = []
    sent = []
    for word in text[0].split():
        elem = word.split(UNDER_BAR)
        assert len(elem) == 2

        word = elem[0].lower()
        tag = elem[1].decode(file_encoding)
        syn = ''
        ne = ''
        prd = VERB if tag.startswith(VERB) else SLASH
        prop = []
        sent.append((word, tag, syn, ne, prd, prop))
    corpus.append(sent)
    return corpus


def get_id_corpus(corpus, vocab_word):
    """
    :param corpus: 1D: n_sents, 2D: n_words; elem=(form, tag, syn, ne, prd, prop)
    :param vocab_word: Vocab()
    :return: 1D: n_sents, 2D: n_words; elem=word id
    """
    return [get_word_ids(sent, vocab_word) for sent in corpus]


def get_word_ids(sent, vocab_word):
    """
    :param sent: 1D: n_words; elem=(form, tag, syn, ne, prd, prop)
    :param vocab_word: Vocab()
    :return: 1D: n_words; elem=word id
    """
    word_ids = []
    for w_index, w in enumerate(sent):
        w_id = vocab_word.get_id(w[0])
        if w_id is None:
            w_id = vocab_word.get_id(UNK)
        word_ids.append(w_id)
    return word_ids


def get_x(corpus, vocab_word):
    """
    :param corpus: 1D: n_sents, 2D: n_words; elem=(form, tag, syn, ne, prd, prop)
    :param vocab_word: Vocab()
    :return: samples: 1D: n_sents, 2D: n_prds; elem=(ctx, marks, labels)
    """
    samples = []
    for i, sent in enumerate(corpus):
        id_sent = get_word_ids(sent, vocab_word)
        prd_indices = [i for i, w in enumerate(sent) if w[4] != '-']

        sample = []
        for prd_i, prd_index in enumerate(prd_indices):
            prd_id = id_sent[prd_index]
            ctx = get_context(id_sent, prd_index)
            ctx = [[w_id, prd_id] + ctx for w_id in id_sent]
            marks = get_marks(id_sent, prd_index)

            assert len(ctx) == len(marks)
            sample.append((ctx, marks))

        samples.append(sample)

    return samples


def get_y(corpus, vocab_label):
    y = []
    for i, sent in enumerate(corpus):
        prd_indices = [i for i, w in enumerate(sent) if w[4] != '-']

        labels = []
        for prd_index in xrange(len(prd_indices)):
            label_ids = []
            for label in _get_labels(prd_index, sent):
                if vocab_label.has_key(label):
                    label_id = vocab_label.get_id(label)
                else:
                    label_id = vocab_label.get_id('O')
                label_ids.append(label_id)
            labels.append(label_ids)
        y.append(labels)

    return y


def get_vocab_label(corpus, vocab_label_tmp=None, cut_label=0):
    iob_labels = _get_iob_labels(corpus)
    cnt = Counter(_get_label_set(iob_labels))
    labels = [(w, c) for w, c in sorted(cnt.iteritems(), key=lambda x: x[1], reverse=True) if c > cut_label]
    say(str(labels))
    return _create_vocab_label(vocab_label_tmp, iob_labels, labels)


def _create_vocab_label(vocab_label_tmp, iob_labels, labels):
    if vocab_label_tmp:
        vocab_label = deepcopy(vocab_label_tmp)
    else:
        vocab_label = Vocab()
        vocab_label.add_word('O')

    labels = [label for label, count in labels]
    for label in iob_labels:
        if label == 'O':
            continue
        if label[2:] in labels:
            vocab_label.add_word(label)

    return vocab_label


def _get_iob_labels(corpus):
    labels = []
    for i, sent in enumerate(corpus):
        prd_indices = [i for i, w in enumerate(sent) if w[4] != '-']
        labels.extend(label for prd_index in xrange(len(prd_indices)) for label in _get_labels(prd_index, sent))
    return labels


def _get_label_set(labels):
    label_set = []
    for label in labels:
        if label == 'O':
            label_set.append(label)
        else:
            label_set.append(label[2:])
    return label_set


def concat_x_y(x, y):
    assert len(x) == len(y), '%d %d' % (len(x), len(y))
    xy = []
    for sent_x, sent_y in zip(x, y):
        assert len(sent_x) == len(sent_y)
        sent_xy = []
        for prd_x, prd_y in zip(sent_x, sent_y):
            assert len(prd_x[0]) == len(prd_y), '%d %d' % (len(prd_x), len(prd_y))
            sent_xy.append((prd_x[0], prd_x[1], prd_y))
        xy.append(sent_xy)
    return xy


def get_context(sent, prd_index, window=5):
    slide = window / 2
    prd_index += slide
    pad = [0 for i in xrange(slide)]
    padded_sent = pad + sent + pad

    return padded_sent[prd_index - slide: prd_index + slide + 1]


def get_marks(sent, prd_index, window=5):
    marks = []
    slide = window / 2

    prev = prd_index - slide
    pro = prd_index + slide
    for i, w in enumerate(sent):
        if prev <= i <= pro:
            marks.append(1.0)
        else:
            marks.append(0.0)

    return marks


def _get_labels(prd_i, sent):
    labels = []
    prev = None
    for w in sent:
        arg = w[5][prd_i]
        if arg.startswith('('):
            if arg.endswith(')'):
                prev = arg.split("*")[0][1:]
                label = 'B-' + prev
                prev = None
            else:
                prev = arg[1:-1]
                label = 'B-' + prev
        else:
            if prev:
                label = 'I-' + prev
                if arg.endswith(')'):
                    prev = None
            else:
                label = 'O'
        labels.append(label)
    return labels


def get_samples(phi_sets, emb):
    """
    :param phi_sets: samples: 1D: n_sents, 2D: n_prds; elem=(ctx, marks, labels)
    :param emb:
    :return: 1D: n_samples, 2D: (x, y)
    """
    samples = []
    for phi in phi_sets:
        # 1D: n_prds, 2D: n_words
        for ctx, mark, label in phi:
            x = array(get_phi_vecs(ctx, mark, emb), is_float=True)
            y = array(label)

            if len(y) < 2:
                continue

            assert len(x) == len(y), '\n%s\n%s\n' % (str(x), str(y))
            samples.append((x, y))

    return samples


def get_sample_x(phi_sets, emb):
    samples = []
    for phi in phi_sets:
        # 1D: n_prds, 2D: n_words
        for ctx, mark in phi:
            x = array(get_phi_vecs(ctx, mark, emb), True)
            samples.append(x)
    return samples


def get_phi_vecs(ctx, mark, emb):
    """
    :param ctx: 1D: n_words, 2D: window + 1; elem=word id
    :param mark: 1D: n_words; elem=float (1.0 or 0.0)
    :param emb: 1D: n_vocab, 2D: dim_emb
    :return: 1D: n_words, 2D: (window + 1) * dim_w + 1
    """
    phi_vecs = []
    for c, m in zip(ctx, mark):
        vec = []
        for w_id in c:
            vec.extend(emb[w_id])
        vec.append(m)
        phi_vecs.append(vec)
    return phi_vecs


def get_batches(samples, batch_size):
    """
    :param samples: 1D: n_samples, 2D: (sample_x, sample_y); 3D: n_words
    :return shared_sampleset: 1D: 8, 2D: n_baches, 3D: batch
    """

    np.random.shuffle(samples)
    samples.sort(key=lambda sample: len(sample[0]))  # sort with n_words

    ##############
    # Initialize #
    ##############
    batches = []
    batch_x = []
    batch_y = []
    prev_n_words = len(samples[0][0])

    for sample in samples:
        sample_x, sample_y = sample
        n_words = len(sample_y)

        ###############
        # Add a batch #
        ###############
        if len(batch_x) == batch_size or prev_n_words != n_words:
            assert len(batch_x) == len(batch_y)
            batches.append((array(batch_x, True), array(batch_y)))
            prev_n_words = n_words
            batch_x = []
            batch_y = []

        ##################
        # Create a batch #
        ##################
        batch_x.append(sample_x)
        batch_y.append(sample_y)

    if batch_x:
        assert len(batch_x) == len(batch_y)
        batches.append((array(batch_x, True), array(batch_y)))

    return batches


def array(_sample, is_float=False):
    if is_float:
        return np.asarray(_sample, dtype=theano.config.floatX)
    return np.asarray(_sample, dtype='int32')
