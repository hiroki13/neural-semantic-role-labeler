import sys
from collections import defaultdict
from copy import deepcopy
import gzip
import cPickle

import numpy as np
import theano

from ..ling.vocab import Vocab, PAD, UNK


def say(s, stream=sys.stdout):
    stream.write(s + '\n')
    stream.flush()


def load_conll(path, exclude=False, file_encoding='utf-8'):
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
                if exclude and (len(sent[0][5]) == 0 or len(sent) < 2):
                    pass
                else:
                    corpus.append(sent)
                sent = []

        if sent:
            corpus.append(sent)

    return corpus


def load_init_emb(init_emb):
    vocab = Vocab()
    vocab.add_word(PAD)
    vocab.add_word(UNK)

    vec = {}
    with open(init_emb) as f_words:
        for line in f_words:
            line = line.strip().decode('utf-8').split()
            w = line[0]

            if w[1:-1] == UNK:
                w = UNK

            vocab.add_word(w)
            vec[vocab.get_id(w)] = np.asarray(line[1:], dtype=theano.config.floatX)

    dim = len(line[1:])

    if vec.get(PAD) is None:
        vec[vocab.get_id(PAD)] = np.zeros(dim, dtype=theano.config.floatX)

    emb = [[] for i in xrange(vocab.size())]
    for k, v in vec.items():
        emb[k] = v

    if vec.get(UNK) is None:
        """ averaging """
        avg = np.zeros(dim, dtype=theano.config.floatX)
        emb[vocab.get_id(UNK)] = avg
        for i in xrange(len(emb)):
            avg += emb[i]
        avg_vec = avg / len(emb)
        emb[vocab.get_id(UNK)] = avg_vec

    emb = np.asarray(emb, dtype=theano.config.floatX)

    assert emb.shape[0] == vocab.size()

    return emb, vocab


def set_args(corpus):
    args = []
    arg_dict = Vocab()
    n_args = 0

    for sent in corpus:
        n_prds = len(sent[0][5])
        sent_args = [[] for i in xrange(n_prds)]
        if n_prds == 0:
            args.append(sent_args)
            continue
        for prd_i in xrange(n_prds):
            prev = None
            for w in sent:
                arg = w[5][prd_i]
                if arg.startswith('('):
                    if arg[1] != 'C' and arg[1] != 'V':
                        n_args += 1
                    if arg.endswith(')'):
                        prev = arg[1:-2]
                        arg_label = 'B-' + prev
                        arg_dict.add_word(arg_label)
                        sent_args[prd_i].append(arg_dict.get_id(arg_label))
                        prev = None
                    else:
                        prev = arg[1:-1]
                        arg_label = 'B-' + prev
                        arg_dict.add_word(arg_label)
                        sent_args[prd_i].append(arg_dict.get_id(arg_label))
                else:
                    if prev:
                        arg_label = 'I-' + prev
                        arg_dict.add_word(arg_label)
                        sent_args[prd_i].append(arg_dict.get_id(arg_label))
                        if arg.endswith(')'):
                            prev = None
                    else:
                        arg_label = 'O'
                        arg_dict.add_word(arg_label)
                        sent_args[prd_i].append(arg_dict.get_id(arg_label))

        args.append(sent_args)
    return args, arg_dict


def set_test_args(corpus, arg_dict):
    args = []
    test_arg_dict = deepcopy(arg_dict)
    for sent in corpus:
        n_prds = len(sent[0][5])
        sent_args = [[] for i in xrange(n_prds)]
        if n_prds == 0:
            args.append(sent_args)
            continue
        for prd_i in xrange(n_prds):
            prev = None
            for w in sent:
                arg = w[5][prd_i]
                if arg.startswith('('):
                    if arg.endswith(')'):
                        prev = arg[1:-2]
                        arg_label = 'B-' + prev
                        if arg_dict.has_key(arg_label):
                            sent_args[prd_i].append(arg_dict.get_id(arg_label))
                        else:
                            sent_args[prd_i].append(arg_dict.get_id('O'))
                            test_arg_dict.add_word(arg_label)
                        prev = None
                    else:
                        prev = arg[1:-1]
                        arg_label = 'B-' + prev
                        if arg_dict.has_key(arg_label):
                            sent_args[prd_i].append(arg_dict.get_id(arg_label))
                        else:
                            sent_args[prd_i].append(arg_dict.get_id('O'))
                            test_arg_dict.add_word(arg_label)
                else:
                    if prev:
                        arg_label = 'I-' + prev
                        if arg_dict.has_key(arg_label):
                            sent_args[prd_i].append(arg_dict.get_id(arg_label))
                        else:
                            sent_args[prd_i].append(arg_dict.get_id('O'))
                            test_arg_dict.add_word(arg_label)
                        if arg.endswith(')'):
                            prev = None
                    else:
                        arg_label = 'O'
                        sent_args[prd_i].append(arg_dict.get_id(arg_label))
        args.append(sent_args)
    return args, test_arg_dict


def print_args(data, file_encoding='utf-8'):
    arg_dict = defaultdict(lambda: len(arg_dict))
    corpus = []
    with open(data) as f:
        sent = []
        for line in f:
            es = line.rstrip().split()
            if len(es) > 1:
                word = es[0].decode(file_encoding)
                tag  = es[1].decode(file_encoding)
                syn  = es[2].decode(file_encoding)
                ne   = es[3].decode(file_encoding)
                prd  = es[4].decode(file_encoding)
                prop = []

                if len(es) > 5:
                    prop = es[5:]
                sent.append((word, tag, syn, ne, prd, prop))
            else:
                # reached end of sentence
                corpus.append(sent)
                sent = []
        if sent:
            corpus.append(sent)

    for sent in corpus:
        n_prds = len(sent[0][5])
        sent_args = [[] for i in xrange(n_prds)]
        for prd_i in xrange(n_prds):
            prev = None
            for w in sent:
                arg = w[5][prd_i]
                if arg.startswith('('):
                    if arg.endswith(')'):
                        prev = arg[1:-2]
                        arg_label = 'B-' + prev
                        sent_args[prd_i].append(arg_label)
                        prev = None
                    else:
                        prev = arg[1:-1]
                        arg_label = 'B-' + prev
                        sent_args[prd_i].append(arg_label)
                else:
                    if prev:
                        arg_label = 'I-' + prev
                        sent_args[prd_i].append(arg_label)
                        if arg.endswith(')'):
                            prev = None
                    else:
                        arg_label = 'O'
                        sent_args[prd_i].append(arg_label)
        for w_i, w in enumerate(sent):
            arg = ''
            for p_i in xrange(n_prds):
                arg += sent_args[p_i][w_i] + '\t'
            print '%s\t%s\t%s\t%s\t%s\t%s' % (w[0], w[1], w[2], w[3], w[4], arg)
        print


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


def output_results(corpus, prd_indices, arg_dict, predicts, path):
    def get_spans(pred_args):
        spans = []
        span = []
        for w_i, a_id in enumerate(pred_args):
            label = arg_dict.get_word(a_id)
            if label.startswith('B-'):
                if span:
                    spans.append(span)
                span = [label[2:], w_i, w_i]
            elif label.startswith('I-'):
                if span:
                    if label[2:] == span[0]:
                        span[2] = w_i
                    else:
                        spans.append(span)
                        span = [label[2:], w_i, w_i]
                else:
                    span = [label[2:], w_i, w_i]
            else:
                if span:
                    spans.append(span)
                span = []
        if span:
            spans.append(span)
        return spans

    def convert(sent, spans):
        k = 0
        args = []
        for w_i in xrange(len(sent)):
            if k >= len(spans):
                args.append('*')
                continue
            span = spans[k]
            if span[1] < w_i < span[2]:  # within span
                args.append('*')
            elif w_i == span[1] and w_i == span[2]:  # begin and end of span
                args.append('(' + span[0] + '*)')
                k += 1
            elif w_i == span[1]:  # begin of span
                args.append('(' + span[0] + '*')
            elif w_i == span[2]:  # end of span
                args.append('*)')
                k += 1
            else:
                args.append('*')  # without span
        return args

    k = 0
    f = open(path, 'w')

    for sent_i in xrange(len(corpus)):
        sent = corpus[sent_i]
        prds = prd_indices[sent_i]
        column = []
        for i in xrange(len(sent)):
            column.append([[] for j in xrange(len(prds) + 1)])
        p_i = 0
        for w_i, w in enumerate(sent):
            column[w_i][0] = w[4]  # base form of prd
            if w_i in prds:
                p_i += 1
                spans = get_spans(predicts[k])
                args = convert(sent, spans)
                for w_j, a_label in enumerate(args):
                    column[w_j][p_i] = a_label
                k += 1
        for c in column:
            text = "\t".join(c)
            print >> f, text
        print >> f


def print_iob_text(corpus, args, arg_dict):
    out = open('iob.txt', 'w')

    for s_i, sent in enumerate(corpus):
        sent_args = args[s_i]
        n_prds = len(sent[0][5])
        for w_i, w in enumerate(sent):
            arg = ''
            for p_i in xrange(n_prds):
                arg += arg_dict.get_word(sent_args[p_i][w_i]) + '\t'
            print >> out, '%s\t%s\t%s\t%s\t%s\t%s' % (w[0], w[1], w[2], w[3], w[4], arg)
        print >> out


def set_args_iob(corpus, arg_dict=None):

    def get_args(sent):
        n_prds = len(sent[0][5])
        sent_args = [[] for i in xrange(n_prds)]
        if n_prds == 0:
            return sent_args

        for prd_i in xrange(n_prds):
            prev = None
            for w in sent:
                arg = w[5][prd_i]
                if arg.startswith('('):
                    if arg.endswith(')'):
                        arg_label = 'S-' + arg[1:-2]
                        a_dict.add_word(arg_label)
                        sent_args[prd_i].append(a_dict.get_id(arg_label))
                        prev = None
                    else:
                        prev = arg[1:-1]
                        arg_label = 'B-' + prev
                        a_dict.add_word(arg_label)
                        sent_args[prd_i].append(a_dict.get_id(arg_label))
                else:
                    if prev:
                        if arg.endswith(')'):
                            arg_label = 'E-' + prev
                            prev = None
                        else:
                            arg_label = 'I-' + prev
                        a_dict.add_word(arg_label)
                        sent_args[prd_i].append(a_dict.get_id(arg_label))
                    else:
                        arg_label = 'O'
                        a_dict.add_word(arg_label)
                        sent_args[prd_i].append(a_dict.get_id(arg_label))
        return sent_args

    if arg_dict:
        a_dict = deepcopy(arg_dict)
    else:
        a_dict = Vocab()
    args = [get_args(sentence) for sentence in corpus]

    return args, a_dict


def output_results_iob(corpus, prd_indices, arg_dict, predicts, path):
    def get_spans(pred_args):
        spans = []
        span = []
        for w_i, a_id in enumerate(pred_args):
            label = arg_dict.get_word(a_id)
            if label.startswith('B-'):
                if span:
                    spans.append(span)
                span = [label[2:], w_i, w_i]
            elif label.startswith('I-'):
                if span:
                    if label[2:] == span[0]:
                        span[2] = w_i
                    else:
                        spans.append(span)
                        span = [label[2:], w_i, w_i]
                else:
                    span = [label[2:], w_i, w_i]
            elif label.startswith('S-'):
                if span:
                    spans.append(span)
                    span = []
                spans.append([label[2:], w_i, w_i])
            elif label.startswith('E-'):
                if span:
                    if label[2:] == span[0]:
                        span[2] = w_i
                        spans.append(span)
                    else:
                        spans.append(span)
                        spans.append([label[2:], w_i, w_i])
                    span = []
                else:
                    span = [label[2:], w_i, w_i]
            else:
                if span:
                    spans.append(span)
                span = []
        if span:
            spans.append(span)
        return spans

    def convert(sent, spans):
        k = 0
        args = []
        for w_i in xrange(len(sent)):
            if k >= len(spans):
                args.append('*')
                continue
            span = spans[k]
            if span[1] < w_i < span[2]:  # within span
                args.append('*')
            elif w_i == span[1] and w_i == span[2]:  # begin and end of span
                args.append('(' + span[0] + '*)')
                k += 1
            elif w_i == span[1]:  # begin of span
                args.append('(' + span[0] + '*')
            elif w_i == span[2]:  # end of span
                args.append('*)')
                k += 1
            else:
                args.append('*')  # without span
        return args


    k = 0
    f = open(path, 'w')

    for sent_i in xrange(len(corpus)):
        sent = corpus[sent_i]
        prds = prd_indices[sent_i]
        column = []
        for i in xrange(len(sent)):
            column.append([[] for j in xrange(len(prds) + 1)])
        p_i = 0
        for w_i, w in enumerate(sent):
            column[w_i][0] = w[4]  # base form of prd
            if w_i in prds:
                p_i += 1
                spans = get_spans(predicts[k])
                args = convert(sent, spans)
                for w_j, a_label in enumerate(args):
                    column[w_j][p_i] = a_label
                k += 1
        for c in column:
            text = "\t".join(c)
            print >> f, text
        print >> f


