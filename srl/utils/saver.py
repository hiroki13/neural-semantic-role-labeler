import gzip
import cPickle


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def write_to_file(text, path='tmp.txt', file_encoding='utf-8'):
    f = open(path, 'w')
    print >> f, text.encode(file_encoding)
    f.close()


def _get_spans(args, vocab_label):
    spans = []
    span = []
    for w_i, a_id in enumerate(args):
        label = vocab_label.get_word(a_id)
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


def _map_span_to_str(sent, spans):
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


def _map_span_to_str_sem(sent, spans):
    k = 0
    args = []
    for w_i in xrange(len(sent)):
        if k >= len(spans):
            args.append('*')
            continue
        span = spans[k]
        if span[1] < w_i < span[2]:  # within span
            args.append('*%s*' % span[0])
        elif w_i == span[1] and w_i == span[2]:  # begin and end of span
            args.append('(' + span[0] + '*)')
            k += 1
        elif w_i == span[1]:  # begin of span
            args.append('(' + span[0] + '*')
        elif w_i == span[2]:  # end of span
            args.append('*' + span[0] + ')')
            k += 1
        else:
            args.append('*')  # without span
    return args


def save_predicted_prop(corpus, vocab_label, predicts, path):
    f = open(path, 'w')
    prd_index = 0
    for sent_i in xrange(len(corpus)):
        sent = corpus[sent_i]
        prds = [i for i, w in enumerate(sent) if w[4] != '-']

        column = []
        for i in xrange(len(sent)):
            column.append([[] for j in xrange(len(prds) + 1)])

        p_i = 0
        for w_i, w in enumerate(sent):
            column[w_i][0] = w[4]  # base form of prd
            if w_i in prds:
                p_i += 1
                spans = _get_spans(predicts[prd_index], vocab_label)
                args = _map_span_to_str(sent, spans)
                for w_j, a_label in enumerate(args):
                    column[w_j][p_i] = a_label
                prd_index += 1
        for c in column:
            text = "\t".join(c)
            print >> f, text
        print >> f
    f.close()


def save_predicted_srl(corpus, vocab_label, predicts, path, file_encoding='utf-8'):
    f = open(path, 'w')
    k = 0
    for sent_i in xrange(len(corpus)):
        sent = corpus[sent_i]
        prds = [i for i, w in enumerate(sent) if w[4] != '-']

        column = []
        for i in xrange(len(sent)):
            column.append([[] for j in xrange(len(prds) + 2)])

        p_i = 1
        for w_i, w in enumerate(sent):
            column[w_i][0] = w[0]  # base form of prd
            column[w_i][1] = w[4]  # base form of prd
            if w_i in prds:
                p_i += 1
                spans = _get_spans(predicts[k], vocab_label)
                args = _map_span_to_str(sent, spans)
                for w_j, a_label in enumerate(args):
                    column[w_j][p_i] = a_label
                k += 1
        for c in column:
            text = "\t".join(c)
            print >> f, text.encode(file_encoding)
        print >> f
    f.close()


def output_predicted_srl_to_cmd(corpus, vocab_label, predicts, file_encoding='utf-8'):
    k = 0
    for sent_i in xrange(len(corpus)):
        sent = corpus[sent_i]
        prds = [i for i, w in enumerate(sent) if w[4] != '-']

        column = []
        for i in xrange(len(sent)):
            column.append([[] for j in xrange(len(prds) + 2)])

        p_i = 1
        for w_i, w in enumerate(sent):
            column[w_i][0] = w[0]  # base form of prd
            column[w_i][1] = w[4]  # base form of prd
            if w_i in prds:
                p_i += 1
                spans = _get_spans(predicts[k], vocab_label)
                args = _map_span_to_str_sem(sent, spans)
                for w_j, a_label in enumerate(args):
                    column[w_j][p_i] = a_label
                k += 1
        for c in column:
            text = "\t".join(c)
            print text.encode(file_encoding)
        print


