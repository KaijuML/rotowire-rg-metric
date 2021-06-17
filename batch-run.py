"""
Use this script to run inference on several gens at the same time. You can
dispatch computations on several gpus or run them all on one.

Note that a number of assumptions are made because it's easier this way.
More flexibility can be added: issues and/or discussions are welcome!

In particular: this script only takes rotowire-folder as input, which is assumed
to have the following structure:

 - `models` where trained RG models are stored ([pretrained models](https://dl.orangedox.com/rg-models)).
 - `output` where everything created by the script is stored: vocabularies,
   training examples, extracted list of mentions, etc.
 - `gens`  you wish to evaluate all the generated texts found here
"""
from data_utils import prep_generated_data
from run import main as inference_run_main
from utils import Container, grouped

from argparse import ArgumentParser

import multiprocessing as mp
import os


def get_parser():
    parser = ArgumentParser(description='Use Information Extractor models, '
                                        'on Rotowire data. This script supports'
                                        ' usage in parallel runs.')

    group = parser.add_argument_group('Script behavior')
    group.add_argument('--test', dest='test', default=False,
                       action="store_true",
                       help='use test data instead of validation data')
    group.add_argument('--show-correctness', dest="show_correctness",
                       action='store_true', help="When doing inference, add a "
                                                 "sign |RIGHT or |WRONG to "
                                                 "generated tuples")

    group = parser.add_argument_group('File system')
    group.add_argument('--rotowire-folder', dest='rotowire_folder', required=True)
    group.add_argument('--vocab-prefix', dest='vocab_prefix', default='',
                       help='prefix of .dict and .labels files')

    group = parser.add_argument_group('Evaluation options')
    group.add_argument('--batch-size', dest='batch_size', default=32, type=int,
                       help='batch size')
    group.add_argument('--ignore-idx', dest='ignore_idx', default=None, type=int,
                       help="The index of NONE label in your .label file")
    group.add_argument('--average-func', dest='average_func', default='arithmetic',
                       choices=['geometric', 'arithmetic'],
                       help='Use geometric/arithmetic mean to ensemble models')

    group = parser.add_argument_group('GPUs options')
    group.add_argument('--gpus', dest='gpus', type=int, nargs='+')
    group.add_argument('--ckpts-per-gpu', dest='ckpts_per_gpu', type=int,
                       default=1, help="Number of runs on the same gpu")
    group.add_argument('--seed', dest='seed', default=3435, type=int,
                       help='Random seed')

    return parser


def build_container(args, gen_filename, gpu):

    print('Building container')

    # remove .txt from gen_filename and add .h5
    filename_pfx = gen_filename[:-4]

    return Container(
        # args shared by both steps
        test=args.test,
        vocab_prefix=os.path.join(args.rotowire_folder, 'output', args.vocab_prefix),

        # Args for the first step (i.e. running data_utils.py)
        gen_fi=os.path.join(args.rotowire_folder, 'gens', gen_filename),
        output_fi=os.path.join(args.rotowire_folder, 'output', filename_pfx+'.h5'),
        input_path=os.path.join(args.rotowire_folder, 'json'),

        # args for the second step (i.e. running run.py)
        just_eval=True,
        datafile=os.path.join(args.rotowire_folder, 'output', args.vocab_prefix + '.h5'),
        preddata=os.path.join(args.rotowire_folder, 'output', filename_pfx+'.h5'),
        eval_models=os.path.join(args.rotowire_folder, 'models'),
        gpu=gpu,
        ignore_idx=args.ignore_idx,
        batch_size=args.batch_size,
        average_func=args.average_func,
        show_correctness=args.show_correctness,
        store_results=os.path.join(args.rotowire_folder, filename_pfx + '.json'),
        seed=args.seed
    )


def single_main(args):

    gen_fi = args.pop('gen_fi')
    dict_pfx = args.vocab_prefix  # don't pop this one
    output_fi = args.pop('output_fi')
    input_path = args.pop('input_path')

    # Run data_utils.py in -mode prep_gen_data
    prep_generated_data(gen_fi, dict_pfx, output_fi, path=input_path, test=args.test)

    # Run run.py
    results = inference_run_main(args.to_namespace())

    print(results.__dict__)


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()

    gens_folder = os.path.join(args.rotowire_folder, "gens")
    gens = [filename
            for filename in os.listdir(gens_folder)
            if filename.endswith('.txt')]

    group_size = len(args.gpus) * args.ckpts_per_gpu

    for grouped_gens in grouped(gens, group_size):

        # We build a list of containers, one for each gen.
        # Checkpoints are dispatched to gpus, each gpu handling
        # 'args.ckpts_per_gpu' checkpoints.

        _gpus = [g for g in args.gpus for _ in range(args.ckpts_per_gpu)]
        containers = [
            build_container(args, gen, gpu)
            for gen, gpu in zip(grouped_gens, _gpus)
            if gen is not None
        ]

        processes = [mp.Process(target=single_main, args=(container,))
                     for container in containers]
        [p.start() for p in processes]
        [p.join() for p in processes]


if __name__ == '__main__':
    main()
