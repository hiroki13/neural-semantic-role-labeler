from copy import deepcopy

import numpy as np
import theano

from ..ling.vocab import Vocab, UNK


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


def get_phi(corpus, vocab_word, a_dict=None):
    """
    :param corpus: 1D: n_sents, 2D: n_words; elem=(form, tag, syn, ne, prd, prop)
    :param vocab_word: Vocab()
    :param a_dict: Vocab()
    :return: samples: 1D: n_sents, 2D: n_prds; elem=(ctx, marks, labels)
    """
    if a_dict:
        label_dict = deepcopy(a_dict)
    else:
        label_dict = Vocab()

    samples = []
    for i, sent in enumerate(corpus):
        id_sent = get_word_ids(sent, vocab_word)
        prd_indices = [i for i, w in enumerate(sent) if w[4] != '-']
        assert len(prd_indices) == len(sent[0][5])

        sample = []
        for prd_i, prd_index in enumerate(prd_indices):
            prd_id = id_sent[prd_index]
            ctx = get_context(id_sent, prd_index)
            ctx = [ctx + [w_id, prd_id] for w_id in id_sent]
            marks = get_marks(id_sent, prd_index)
            labels = get_labels(prd_i, sent, label_dict)

            assert len(ctx) == len(marks) == len(labels)
            sample.append((ctx, marks, labels))

        samples.append(sample)

    return samples, label_dict


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


def get_labels(prd_i, sent, arg_dict):
    sent_args = []
    prev = None
    for w in sent:
        arg = w[5][prd_i]
        if arg.startswith('('):
            if arg.endswith(')'):
                prev = arg[1:-2]
                arg_label = 'B-' + prev
                arg_dict.add_word(arg_label)
                sent_args.append(arg_dict.get_id(arg_label))
                prev = None
            else:
                prev = arg[1:-1]
                arg_label = 'B-' + prev
                arg_dict.add_word(arg_label)
                sent_args.append(arg_dict.get_id(arg_label))
        else:
            if prev:
                arg_label = 'I-' + prev
                arg_dict.add_word(arg_label)
                sent_args.append(arg_dict.get_id(arg_label))
                if arg.endswith(')'):
                    prev = None
            else:
                arg_label = 'O'
                arg_dict.add_word(arg_label)
                sent_args.append(arg_dict.get_id(arg_label))
    return sent_args


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
            x = array(get_phi_vecs(ctx, mark, emb), True)
            y = array(label)

            assert len(x) == len(y), '\n%s\n%s\n' % (str(x), str(y))
            samples.append((x, y))

    return samples


def get_batches(samples, batch_size, test=False):
    """
    :param samples: 1D: n_samples, 2D: (sample_x, sample_y); 3D: n_words
    :return shared_sampleset: 1D: 8, 2D: n_baches, 3D: batch
    """

    if samples is None:
        return None

    if test is False:
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
    else:
        return np.asarray(_sample, dtype='int32')


def get_phi_vecs(ctx, mark, emb):
    """
    :param ctx: 1D: n_words, 2D: window + 1; elem=word id
    :param mark: 1D: n_words; elem=float (1.0 or 0.0)
    :param emb:
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


def convert_data_test(id_sents, prds, id_ctx, marks, args, emb):
    batch_x = []
    batch_y = []

    for s_i in xrange(len(id_sents)):
        sent_w = [emb[w_id] for w_id in id_sents[s_i]]
        sent_prds = prds[s_i]
        sent_ctx = id_ctx[s_i]
        sent_marks = marks[s_i]
        sent_args = args[s_i]

        for p_i, p_index in enumerate(sent_prds):
            prd = sent_w[p_index]
            ctx = []
            for w_index in sent_ctx[p_i]:
                ctx.extend(sent_w[w_index])

            mark = sent_marks[p_i]
            arg = sent_args[p_i]

            sent_sample = []
            for w_index, w in enumerate(sent_w):
                sample = []
                sample.extend(w)
                sample.extend(prd)
                sample.extend(ctx)
                sample.append(mark[w_index])
                sent_sample.append(sample)

            batch_x.append(np.asarray([sent_sample], dtype=theano.config.floatX))
            batch_y.append(np.asarray([arg], dtype='int32'))

    return batch_x, batch_y


def shuffle_batches(batches,  batch):
    """
    :param batches: 1D: n_batches, 2D: batch_size; (sample_x, sample_y)
    :param batch:
    :return:
    """
    samples = [(sample_x, sample_y) for batch in batches for sample_x, sample_y in zip(*batch)]
    return get_batches(samples, batch)


def shuffle(sample_x, sample_y):
    new_x = []
    new_y = []

    indices = [i for i in xrange(len(sample_x))]
    np.random.shuffle(indices)

    for i in indices:
        batch_x = sample_x[i]
        batch_y = sample_y[i]
        b_indices = [j for j in xrange(len(batch_x))]
        np.random.shuffle(b_indices)
        new_x.append([batch_x[j] for j in b_indices])
        new_y.append([batch_y[j] for j in b_indices])

    return new_x, new_y
