from argparse import ArgumentParser

from models import ConvRgModel, RecurrentRgModel
from sampler import build_dataset_iter
from data import prep_data
from trainer import Trainer


def get_parser():
    parser = ArgumentParser(description='Train/Use an Information Extractor '
                            'model, on Rotowire data.')

    group = parser.add_argument_group('Script behavior')
    group.add_argument('--just-eval', dest='just_eval', default=False,
                       action="store_true", help='just run evaluation script')
    group.add_argument('--test', dest='test', default=False,
                       action="store_true", help='use test data')

    group = parser.add_argument_group('File system')
    group.add_argument('--datafile', dest='datafile',
                       help='path to hdf5 file containing train/val data')
    group.add_argument('--preddata', dest='preddata', default='',
                       help='path to hdf5 file containing candidate relations '
                            'from generated data')
    group.add_argument('--savefile', dest='savefile', default='',
                       help='path to a directory where to model should be saved')
    group.add_argument('--eval-models', dest='eval_models', default='',
                       help='path to trained extractor models')
    group.add_argument('--dict-pfx', dest='dict_pfx', default='',
                       help='prefix of .dict and .labels files')

    group = parser.add_argument_group('Evaluation options')
    group.add_argument('--geom', dest='geom', default=False,
                       action="store_true", help='average models geometrically')

    group = parser.add_argument_group('Training options')
    group.add_argument('--epochs', dest='epochs', default=10, type=int,
                       help='Number of training epochs')
    group.add_argument('--gpu', dest='gpu', default=1, type=int, help='gpu idx')
    group.add_argument('--batch-size', dest='batch_size', default=32, type=int,
                       help='batch size')
    group.add_argument('--lr', dest='lr', default=0.7, type=float,
                       help='learning rate')
    group.add_argument('--lr-decay', dest='lr_decay', default=0.5, type=float,
                       help='decay factor')
    group.add_argument('--clip', dest='clip', default=5,
                       help='clip grads so they do not exceed this')
    group.add_argument('--seed', dest='seed', default=3435, type=int,
                       help='Random seed')

    group = parser.add_argument_group('Model configuration')
    group.add_argument('--model', dest='model', choices=['lstm', 'conv'])
    group.add_argument('--embedding-size', dest='embedding_size', type=int,
                       default=200, help="Dimensions of embedding space")
    group.add_argument('--hidden-dim', dest='hidden_dim', default=500, type=int,
                       help="Hidden dimension of the model")
    group.add_argument('--dropout', dest='dropout', default=0.5, type=float,
                       help='if >0 use dropout regularization')

    group = parser.add_argument_group('Conv specific configuration')
    group.add_argument('--num-filters', dest='num_filters', default=200,
                       type=int, help='number of convolutional filters')

    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()

    datasets, min_dists, vocab_sizes, nlabels = prep_data(args.datafile,
                                                          args.preddata,
                                                          args.test,
                                                          args.just_eval)
    train, val, test = datasets

    datakwargs = {'batch_size': args.batch_size, 'vocab_sizes': vocab_sizes}
    loaders = [
        build_dataset_iter(train, **datakwargs),
        build_dataset_iter(val, **datakwargs, is_eval=True),
        build_dataset_iter(test, **datakwargs, is_eval=True)
    ]

    emb_sizes = [args.embedding_size,
                 args.embedding_size // 2,
                 args.embedding_size // 2]
    vocab_sizes = [p + 1 for p in vocab_sizes]

    if args.just_eval:
        raise RuntimeError('No puedo just_eval right now')

    # Building models
    if args.model == 'lstm':
        model = RecurrentRgModel(vocab_sizes=vocab_sizes,
                                 emb_sizes=emb_sizes,
                                 hidden_dim=args.hidden_dim,
                                 nlabels=nlabels,
                                 dropout=args.dropout)
        model.count_parameters(module_names=['embeddings', 'rnn', 'linear'])
    else:
        model = ConvRgModel(vocab_sizes=vocab_sizes,
                            emb_sizes=emb_sizes,
                            num_filters=args.num_filters,
                            hidden_dim=args.hidden_dim,
                            nlabels=nlabels,
                            dropout=args.dropout)
        model.count_parameters(module_names=['embeddings', 'convolutions', 'linear'])

    trainer = Trainer()

    return model, loaders


if __name__ == '__main__':
    main()
