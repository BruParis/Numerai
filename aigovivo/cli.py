import click

from .data_analysis import generate_cross_corr, ft_selection, model_ft_imp
from .strat import make_new_strat, make_aggr_dict
from .clustering import clustering, simple_era_clustering
from .format_data import data_setup, split_data_clusters
from .prediction import make_prediction, cluster_proba, neutralize_pred, upload_results, compute_predict, cl_pred_diagnostics, pred_diagnostics, optim_proc
from .train import generate_cl_interpo, generate_cl_model, generate_fst_layer_model, generate_snd_layer_model
from .models import generate_model_dict
from .common import *


def models_from_arg(models):
    m_split = models.split(',')
    for m in m_split:
        if m not in MODEL_DICT.keys():
            print('{} is not a refering to any model.'.format(m))
            exit(0)

    model_types = [ModelType[MODEL_DICT[m]] for m in m_split]

    return model_types


@click.group()
def cli():
    return


@cli.command('setup')
def setup():
    data_setup()


@cli.command('corr')
def corr():
    generate_cross_corr()


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


@cli.command('model')
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.argument('folder', type=click.Path(exists=True))
def model(layer, folder):
    generate_model_dict(folder, layer)


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
              default=False,
              show_default=True,
              is_flag=True)
@click.option("-i", "--ft-imp", default=False, show_default=True, is_flag=True)
@click.option("-rs",
              "--random-search",
              default=False,
              show_default=True,
              is_flag=True)
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.option("-c", "--cluster", default=None, show_default=True)
@click.argument('models', default="nn")
@click.argument('folder', type=click.Path(exists=True))
def train(debug, metrics, no_save, ft_imp, random_search, layer, cluster,
          models, folder):

    model_types = models_from_arg(models)
    print('model_types: ', model_types)

    save = not no_save
    if layer == '0':
        if cluster is None:
            print("cluster name not provided")
            return
        generate_cl_model(folder, cluster, model_types, random_search, ft_imp,
                          debug, metrics, save)
        return
    if cluster is not None:
        print("specifying a cluster is unecessary when training a layer")

    if layer == 'fst':
        generate_fst_layer_model(folder, model_types, random_search, ft_imp,
                                 debug, metrics, save)
    if layer == 'snd':
        generate_snd_layer_model(folder, model_types, debug, metrics, save)


@cli.command('aggr')
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.argument('folder', type=click.Path(exists=True))
def aggr(layer, folder):
    make_aggr_dict(layer, folder)


@cli.command('mftimp')
@click.option("-m",
              "--metrics",
              default=False,
              show_default=True,
              is_flag=True)
@click.option("-ns",
              "--no-save",
              default=False,
              show_default=True,
              is_flag=True)
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.option("-c", "--cluster", default=None, show_default=True)
@click.argument('models', default="nn")
@click.argument('folder', type=click.Path(exists=True))
def mftimp(metrics, no_save, layer, cluster, models, folder):

    model_types = models_from_arg(models)
    print('model_types: ', model_types)

    save = not no_save
    if layer == '0':
        if cluster is None:
            print("cluster name not provided")
            return
        model_ft_imp(folder, cluster, model_types, metrics, save)
    else:
        if cluster is not None:
            print("specifying a cluster is unecessary when training a layer")
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
@click.argument('models', default="nn")
@click.argument('folder', type=click.Path(exists=True))
def exec(threadpool, pred, layer, cluster, models, folder):

    model_types = models_from_arg(models)
    print('model_types: ', model_types)

    if pred == 'proba':
        if layer == '0':
            if cluster is None:
                print("cluster name not provided")
                return
            cluster_proba(folder, model_types, cluster)
        else:
            if cluster is not None:
                print(
                    "specifying a cluster is unecessary when training a layer")
            if layer == 'snd':
                print("snd layer doesn't produce proba")
            cluster_proba(folder, model_types)
    elif pred == 'pred':
        make_prediction(folder, layer, model_types)
    elif pred == 'neutralize':
        neutralize_pred(folder, model_types)


@cli.command('optim')
@click.option("-n", "--neutr", default=False, show_default=True, is_flag=True)
@click.argument('models', default="nn")
@click.argument('aggr', default="1")
@click.argument('folder', type=click.Path(exists=True))
def optim(models, aggr, neutr, folder):
    model_types = models_from_arg(models)
    optim_proc(folder, model_types, aggr, neutr)


@cli.command('compute')
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.option("-c", "--cluster", default=None, show_default=True)
@click.argument('models', default="nn")
@click.argument('folder', type=click.Path(exists=True))
def compute(layer, cluster, models, folder):
    model_types = models_from_arg(models)
    print('model_types: ', model_types)

    compute_predict(layer, cluster, folder, model_types)


@cli.command('diag')
@click.option("-l",
              "--layer",
              type=click.Choice(LAYERS, case_sensitive=False),
              default="fst",
              prompt=True)
@click.option('-a', "--aggr-id", default=None)
@click.option('-f', "--ft-n", default=False, show_default=True, is_flag=True)
@click.option("-c", "--cluster", default=None, show_default=True)
@click.argument('folder', type=click.Path(exists=True))
def diag(layer, aggr_id, ft_n, cluster, folder):
    if layer == '0':
        if cluster is None:
            print("cluster name not provided")
            return
        cl_pred_diagnostics(folder, cluster)
    else:
        if cluster is not None:
            print("specifying a cluster is unecessary for layer diagnostic")
        if aggr_id is None:
            print("No aggr pred id for layer diagnostic")
        if layer == 'snd':
            print("snd layer doesn't produce proba")
            return
        pred_diagnostics(folder, aggr_id, ft_n)


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
