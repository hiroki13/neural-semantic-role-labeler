from ..utils.io_utils import say


def count_correct(errors):
    total = 0.0
    correct = 0
    for sent in errors:
        total += len(sent)
        for y_pred in sent:
            if y_pred == 0:
                correct += 1
    return total, correct


def f_measure(predicts, answers, arg_dict):
    def get_spans(sent):
        spans = []
        span = []
        for w_i, a_id in enumerate(sent):
            label = arg_dict.get_word(a_id)
            if label.startswith('B-'):
                if span and span[0][0] != 'V':
                    spans.append(span)
                span = [label[2:], w_i, w_i]
            elif label.startswith('I-'):
                if span:
                    if label[2:] == span[0]:
                        span[2] = w_i
                    else:
                        if span[0][0] != 'V':
                            spans.append(span)
                        span = [label[2:], w_i, w_i]
                else:
                    span = [label[2:], w_i, w_i]
            else:
                if span and span[0][0] != 'V':
                    spans.append(span)
                span = []
        if span:
            spans.append(span)
        return spans

    p_total = 0.
    r_total = 0.
    correct = 0.
    for i in xrange(len(predicts)):
        ys = predicts[i]
        ds = answers[i][0]
        y_spans = get_spans(ys)
        d_spans = get_spans(ds)
        p_total += count_spans(y_spans)
        r_total += count_spans(d_spans)

        for y_span in y_spans:
            if y_span[0].startswith('C'):
                continue
            if y_span in d_spans:
                correct += 1.

    if p_total > 0:
        p = correct / p_total
    else:
        p = 0.
    if r_total > 0:
        r = correct / r_total
    else:
        r = 0.
    if p + r > 0:
        f = (2 * p * r) / (p + r)
    else:
        f = 0.

    say('\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
        f, p, int(correct), int(p_total), r, int(correct), int(r_total)))

    return f


def count_spans(spans):
    total = 0
    for span in spans:
        if not span[0].startswith('C'):
            total += 1
    return total


def show_f1_history(f1_history):
    say('\n\tF1 HISTORY')
    for k, v in sorted(f1_history.items()):
        if len(v) == 2:
            say('\tEPOCH-{:d}  \tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}'.format(k, v[0], v[1]))
        else:
            say('\tEPOCH-{:d}  \tBEST DEV F:{:.2%}'.format(k, v[0]))
    say('\n')
