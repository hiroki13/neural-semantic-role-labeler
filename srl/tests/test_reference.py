import numpy as np
from ..utils.io_utils import say, load_init_emb


def test_emb_reference(argv):
    say('\n\tInitial Embedding Loading...')
    init_emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
    say('\tVocabulary Size: %d' % vocab_word.size())

    print '\ninit_emb is a numpy array'
    print '\tinit_emb: %d' % id(init_emb)
    v1 = get_word_vecs(init_emb)
    print '\tv1: %d ' % id(v1)
    v2 = get_word_vecs(init_emb)
    print '\tv2: %d ' % id(v2)

    print '\ninit_emb is a list that consists of numpy arrays'
    init_emb = [e for e in init_emb]
    print '\tinit_emb: %d' % id(init_emb)
    v1 = get_word_vecs(init_emb)
    print '\tv1: %d ' % id(v1)
    v2 = get_word_vecs(init_emb)
    print '\tv2: %d ' % id(v2)


def get_word_vecs(emb):
    v = emb[0]
    print '\tv in func: %d ' % id(v)
    return v


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train/Test SRL tagger.')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    argv = parser.parse_args()

    test_emb_reference(argv)
