import numpy as np
import theano


def convert_data(id_sents, prds, id_ctx, marks, args, emb):
    sample_x = []
    sample_y = []
    batch_x = []
    batch_y = []

    pre_len = len(id_sents[0])
    for s_i in xrange(len(id_sents)):
        sent_w = [emb[w_id] for w_id in id_sents[s_i]]
        sent_prds = prds[s_i]
        sent_ctx = id_ctx[s_i]
        sent_marks = marks[s_i]
        sent_args = args[s_i]

        """ create prd-per sample """
        p_sample = []
        for p_i, p_index in enumerate(sent_prds):
            prd = sent_w[p_index]
            ctx = []
            for w_index in sent_ctx[p_i]:
                ctx.extend(sent_w[w_index])

            mark = sent_marks[p_i]

            sent_sample = []
            for w_index, w in enumerate(sent_w):
                sample = []
                sample.extend(w)
                sample.extend(prd)
                sample.extend(ctx)
                sample.append(mark[w_index])
                sent_sample.append(sample)
            p_sample.append(sent_sample)

        sent_len = len(sent_w)

        if sent_len == pre_len:
            batch_x.extend(p_sample)
            batch_y.extend(sent_args)
        else:
            sample_x.append(np.asarray(batch_x, dtype=theano.config.floatX))
            sample_y.append(np.asarray(batch_y, dtype='int32'))

            batch_x = []
            batch_x.extend(p_sample)
            batch_y = []
            batch_y.extend(sent_args)

        pre_len = sent_len

    if batch_x:
        sample_x.append(np.asarray(batch_x, dtype=theano.config.floatX))
        sample_y.append(np.asarray(batch_y, dtype='int32'))

    return sample_x, sample_y


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
