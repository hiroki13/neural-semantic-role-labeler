__author__ = 'hiroki'

import theano
import numpy as np

theano.config.floatX = 'float32'
np.random.seed(0)


if __name__ == '__main__':
    import argparse
    import train
    import test

    parser = argparse.ArgumentParser(description='Train/Test SRL tagger.')

    parser.add_argument('-mode', default='train', help='train/test')
    parser.add_argument('--train_data',  help='path to training data')
    parser.add_argument('--dev_data',  help='path to dev data')
    parser.add_argument('--test_data',  help='path to test data')

    # NN architecture
    parser.add_argument('--unit',  default='lstm', help='Unit')
    parser.add_argument('--vocab',  type=int, default=100000000, help='vocabulary size')
    parser.add_argument('--emb',    type=int, default=32,        help='dimension of embeddings')
    parser.add_argument('--window', type=int, default=5,         help='window size for convolution')
    parser.add_argument('--hidden', type=int, default=32,        help='dimension of hidden layer')
    parser.add_argument('--layer',  type=int, default=1,         help='number of layers')

    # training options
    parser.add_argument('--save', type=bool, default=False, help='parameters to be saved or not')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--lr1', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr2', type=float, default=0.01, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0001, help='L2 Reg rate')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='don\'t shuffle training data')
    parser.add_argument('--model', default=None, help='path to model')
    parser.add_argument('--arg_dict', default=None, help='path to arg dict')

    argv = parser.parse_args()

    if argv.mode == 'train':
        train.main(argv)
    else:
        assert argv.model is not None
        assert argv.arg_dict is not None
        test.main(argv)
