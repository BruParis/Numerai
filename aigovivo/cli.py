import click

from .h5 import store_h5
from .data_analysis import compute_corr, ft_selection
from .strat import make_new_strat
from .clustering import clustering, simple_era_clustering
from .format_data import split_data_clusters
from .prediction import make_prediction, cluster_proba, upload_results
from .models import generate_cl_interpo, generate_cl_model, generate_models
from .common import *


@click.group()
def cli():
    return


@cli.command('h5')
def h5():
    store_h5()


@cli.command('corr')
def corr():
    compute_corr()


@cli.command('new')
@click.option(
    "-m",
    "--method",
    type=click.Choice(CLUSTER_METHODS, case_sensitive=False),
    default="cluster",
    prompt=True,
)
@click.argument('folder', type=click.Path(), required=True)
def new(method, folder):
    make_new_strat(method, folder)
    return


@cli.command('ft')
@click.argument('folder', type=click.Path(exists=True))
def ft(folder):
    ft_selection(folder)


@cli.command('cl')
@click.argument('folder', type=click.Path(exists=True))
def cl(folder):
    clustering(folder)
    # elif method == STRAT_ERA_SIMPLE:
    #     simple_era_clustering(folder)
    split_data_clusters(folder)


# TODO : move interpo to train ?
# @cli.command('interpo')
# @click.option("-d", "--debug", default=False, show_default=True, is_flag=True)
# @click.option("-m",
#               "--metrics",
#               default=False,
#               show_default=True,
#               is_flag=True)
# @click.option("-ns",
#               "--no-save",
#               default=True,
#               show_default=True,
#               is_flag=True)
# @click.option("-c", "--cluster", default=None, show_default=True)
# @click.argument('folder', type=click.Path(exists=True))
# def interpo(metrics, debug, no_save, cluster, folder):
#     generate_cl_interpo(folder, metrics, debug, no_save, cluster)


@cli.command('train')
@click.option("-d", "--debug", default=False, show_default=True, is_flag=True)
@click.option("-m",
              "--metrics",
              default=False,
              show_default=True,
              is_flag=True)
@click.option("-ns",
              "--no-save",
              default=True,
              show_default=True,
              is_flag=True)
@click.option("-t",
              "--threadpool",
              default=False,
              show_default=True,
              is_flag=True)
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.option("-c", "--cluster", default=None, show_default=True)
@click.argument('folder', type=click.Path(exists=True))
def train(debug, metrics, no_save, threadpool, layer, cluster, folder):
    save = not no_save
    if layer == '0':
        if cluster is None:
            print("cluster name not provided")
            return
        generate_cl_model(folder, cluster, debug, metrics, save)
    else:
        if cluster is not None:
            print("specifying a cluster is unecessary when training a layer")
        generate_models(folder, layer, debug, metrics, save, threadpool)
    return


@cli.command('exec')
@click.option("-t",
              "--threadpool",
              default=False,
              show_default=True,
              is_flag=True)
@click.option("-p",
              "--pred",
              type=click.Choice(PRED_OPERATIONS, case_sensitive=False),
              default="proba",
              prompt=True)
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.option("-c", "--cluster", default=None, show_default=True)
@click.argument('folder', type=click.Path(exists=True))
def exec(threadpool, pred, layer, cluster, folder):
    if pred == 'proba':
        if layer == '0':
            if cluster is None:
                print("cluster name not provided")
                return
            cluster_proba(folder, cluster)
        else:
            if cluster is not None:
                print(
                    "specifying a cluster is unecessary when training a layer")
            if layer == 'snd':
                print("snd layer doesn't produce proba")
            cluster_proba(folder)
    elif pred == 'prediction':
        make_prediction(folder, layer)


@cli.command('upload')
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.argument('folder', type=click.Path(exists=True))
@click.argument('aggr', type=click.Path(exists=True))
def upload(layer, folder, aggr):
    upload_results(folder, layer, aggr)
