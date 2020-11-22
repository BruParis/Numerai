import argparse
import sys
import time
from collections import deque

from common import *
from reader import set_h5_stores
from prediction import make_prediction, final_pred, validation_score, upload_results

ALL_OPERATIONS = ['set_h5', 'prediction', 'final_prediction', 'upload']


def print_funct_calls(layers, strategies, operations_q):
    l_aux = layers.copy()
    s_aux = strategies.copy()
    o_aux = operations_q.copy()

    while 0 < len(o_aux):
        op = o_aux.popleft()
        print("op: ", op)

        if op == 'set_h5':
            print('   --> set_h5_stores')
            continue

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

                if op == 'upload':
                    print('   --> upload results')
                    break


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

    start_time = time.time()

    while 0 < len(operations_q):
        op = operations_q.popleft()
        print("op: ", op)

        if op == 'set_h5':
            set_h5_stores()
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

                if op == 'upload':
                    upload_results(strat_dir, l)
                    break

    print("--- %s seconds ---" % (time.time() - start_time))

    # while 0 < len(operations_q):
    #     op = operations_q.popleft()
    #     if op == 'upload':

    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
