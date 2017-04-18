import numpy as np
import theano

theano.config.floatX = 'float32'
np.random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deep SRL tagger.')

    parser.add_argument('-mode', default='train', help='train/test')
    parser.add_argument('--data_type',  default='conll', help='conll/pos_tagged')

    parser.add_argument('--train_data',  help='path to training data')
    parser.add_argument('--dev_data',  help='path to dev data')
    parser.add_argument('--test_data',  help='path to test data')

    """ NN architecture """
    parser.add_argument('--unit',  default='gru', help='gru/lstm')
    parser.add_argument('--connect',  default='agg', help='agg/res')
    parser.add_argument('--vocab',  type=int, default=100000000, help='vocabulary size')
    parser.add_argument('--emb',    type=int, default=50,        help='dimension of embeddings')
    parser.add_argument('--window', type=int, default=5,         help='window size for convolution')
    parser.add_argument('--hidden', type=int, default=32,        help='dimension of hidden layer')
    parser.add_argument('--layer',  type=int, default=1,         help='number of layers')

    """ training options """
    parser.add_argument('--cut_label', type=int,  default=0)
    parser.add_argument('--save', type=int, default=0, help='parameters to be saved or not')
    parser.add_argument('--output', type=int, default=0, help='output results to cmd line')
    parser.add_argument('--data_size', type=int, default=1000000, help='data size to be used')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0005, help='L2 Reg rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='don\'t shuffle training data')

    """ test options """
    parser.add_argument('--load_model', default=None, help='path to model')
    parser.add_argument('--load_word', default=None, help='path to words')
    parser.add_argument('--load_label', default=None, help='path to labels')
    parser.add_argument('--load_emb', default=None, help='path to embs')

    argv = parser.parse_args()

    print
    print argv
    print

    if argv.mode == 'train':
        import train
        train.main(argv)
    else:
        import test
        test.main(argv)
