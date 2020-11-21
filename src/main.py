import argparse
import sys
from collections import deque

from common import *
from corr_analysis import feature_era_corr
from clustering import clustering
from data_instruments import split_data_clusters, snd_layer_training_data
from models import generate_models
from prediction import make_prediction, final_pred, validation_score

ALL_OPERATIONS = ['ft_era_corr', 'split_data', 'train',
                  'prediction', 'final_prediction', 'upload']


def print_funct_calls(layers, strategies, operations_q):
    l_aux = layers.copy()
    s_aux = strategies.copy()
    o_aux = operations_q.copy()

    while 0 < len(o_aux):
        op = o_aux.popleft()
        print("op: ", op)

        if op == 'upload':
            break

        for stra in s_aux:
            print(" strat: ", stra)
            for l in l_aux:
                print('  layer: ', l)
                if op == 'prediction':
                    print('   --> make_pred')
                    continue
                if op == 'final_prediction':
                    print('   --> final_pred')
                    print('   --> validation_pred')
                    continue
                if op == 'train':
                    print('   --> generate_models')
                    continue

                if l == 'fst':
                    if op == 'ft_era_corr':
                        print('   --> feature_era_corr')
                        continue
                    if op == 'split_data':
                        if stra == STRAT_CLUSTER:
                            print('   --> clustering')
                        print('   --> split_data_clusters')
                        continue

                elif l == 'snd':
                    if op == 'split_data':
                        print('   --> snd_layer_training_data')
                        continue
                    elif op == 'train':
                        print('   --> generate_models')
                        continue


def main(argv):
    """Program entry point.
    :param argv: command-line arguments
    :type argv: :class:`list`
    """

    arg_parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument("layer",
                            help="<layer> used",
                            nargs=1,
                            default=['fst'])
    # ['fst', 'snd', 'full']
    arg_parser.add_argument("strat",
                            help="<strat> used",
                            nargs=1,
                            default=[STRAT_CLUSTER])
    # [STRAT_ERA_GRAPH, STRAT_CLUSTER, 'full']
    arg_parser.add_argument("operations",
                            help="<operation> to realize",
                            nargs='*',
                            default=['prediction'])
    # ['ft_era_corr', 'split_data', 'train', 'preditction', 'final_prediction', 'upload']

    # arg_parser.add_argument("-f", "--full", action='store_true',
    #                    help="use full dataset")

    args = arg_parser.parse_args(args=argv[1:])
    layers = ['fst', 'snd'] if args.layer[0] == 'full' else args.layer
    strategies = args.strat
    operations_q = deque([
        op for op in ALL_OPERATIONS for arg_op in args.operations if arg_op == op])

    print("layers: ", layers)
    print("strategies: ", strategies)
    print("operations: ", operations_q)

    print_funct_calls(layers, strategies, operations_q)

    while 0 < len(operations_q):
        op = operations_q.popleft()
        print("op: ", op)

        if op == 'upload':
            break

        if op == 'convert':
            # h5_convert()
            continue

        for stra in strategies:
            print("strat: ", stra)
            strat_dir = ERA_CL_DIRNAME if stra == STRAT_CLUSTER else ERA_GRAPH_DIRNAME
            for l in layers:
                if op == 'prediction':
                    make_prediction(stra, l, PREDICTION_TYPES)
                    continue
                if op == 'final_prediction':
                    final_pred(strat_dir)
                    validation_score(strat_dir)
                    continue
                if op == 'train':
                    generate_models(stra, l)
                    continue

                print("layer: ", l)
                if l == 'fst':
                    if op == 'ft_era_corr':
                        feature_era_corr(TRAINING_DATA_FP)
                        continue
                    if op == 'split_data':
                        if stra == STRAT_CLUSTER:
                            clustering(strat_dir)
                        split_data_clusters(strat_dir)
                        continue

                elif l == 'snd':
                    if op == 'split_data':
                        snd_layer_training_data(stra)
                        continue
                    elif op == 'train':
                        generate_models(stra, l)
                        continue

    # while 0 < len(operations_q):
    #     op = operations_q.popleft()
    #     if op == 'upload':

    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
