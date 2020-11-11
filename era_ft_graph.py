import os
import errno
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from feature_order import ft_cat_order_idx

from networkx.algorithms.community import k_clique_communities

MIN_FEATURES = 7

MIN_GRAPH_NODE = 4  # MIN_GRAPH_NODE * eras should make a training dataset large enough
MIN_RATIO_COMMON_FT = 0.6  # 60% of all features in eras subgraph must be common to all

COND_GRAPH_NODE = 8
COND_GRAPH_UNION_FTS = 70

# CORR_THRESHOLD = 0.05
CORR_THRESHOLD = 0.036
# CORR_THRESHOLD = 0.02

# Methods
# 1: Custom
# Modularity maximization :
# python igraph : community_optimal_maximization
# mapEquiation.org infomap
# Algorithme Louvain


def ft_common(fts1, fts2):
    return len(set(fts1) & set(fts2))


def ft_simi(fts1, fts2):
    l_1 = len(fts1)
    l_2 = len(fts2)
    if l_1 == 0 or l_2 == 0:
        return 0

    n_common = len(set(fts1) & set(fts2))
    res = max(n_common/l_1, n_common/l_2)
    return res


def display_graph(g):
    nx.draw(g)
    plt.draw()
    plt.show()


def sum_weight_node(g, n):
    edges_data = [g.get_edge_data(
        *e) for e in g.edges(n)]
    n_w = sum([e['weight'] for e in list(filter(None, edges_data))])

    return n_w


def kclique_graph(g, k):
    com_l = list(k_clique_communities(g, k))

    if len(com_l) < 2:
        return None

    print("kclique done - num comm. found: ", len(com_l))

    com_g_l = [g.subgraph(c).copy() for c in com_l]

    union_com_g = com_g_l[0]
    for com_g in com_g_l[1:]:

        inter_com_g = union_com_g.copy()
        inter_com_g.remove_nodes_from(
            n for n in union_com_g if n not in com_g)

        for node in inter_com_g.nodes:
            node_w_aux = sum_weight_node(union_com_g, node)
            node_w_current = sum_weight_node(com_g, node)

            if node_w_aux > node_w_current:
                com_g.remove_nodes_from([node])
            else:
                union_com_g.remove_nodes_from([node])

        union_com_g = nx.union(union_com_g, com_g)

    return union_com_g


def generate_graph_from_dict(era_dict, display=False):

    g_sim = {e1: {e2: {'weight': ft_simi(fts1, fts2)}
                  for e2, fts2 in era_dict.items()}
             for e1, fts1 in era_dict.items()}

    g_sim = nx.from_dict_of_dicts(g_sim)
    g_sim.remove_edges_from(nx.selfloop_edges(g_sim))

    if display:
        display_graph(g_sim)

    max_w = max(
        map(lambda e_d: e_d[2]['weight'], g_sim.edges.data()))
    print("max_w: ", max_w)

    return g_sim


def inter_eras_features(era_ft_dict, eras):

    ft_l = [ft for e in eras for ft in era_ft_dict[e]]
    set_accu = ft_l[0]
    for ft in ft_l:
        set_accu = set(ft).intersection(set_accu)

    res = list(set_accu)
    return res


def union_eras_features(era_ft_dict, eras):
    ft_l = set([ft for e in eras for ft in era_ft_dict[e]])

    # set() breaks ft order -> need to re-order
    ft_sorted = sorted(ft_l, key=lambda ft: ft_cat_order_idx(ft))

    return ft_sorted


def find_sim_subg(era_ft_dict):
    eras_g_l = []
    min_s = 0.95
    while len(era_ft_dict) > MIN_GRAPH_NODE:
        eras_graph = generate_graph_from_dict(era_ft_dict)

        while min_s > 0.05:
            round_graph = eras_graph.copy()
            round_graph.remove_edges_from(
                [e for e in eras_graph.edges.data() if e[2]['weight'] < min_s])
            round_graph.remove_nodes_from(
                list(nx.isolates(round_graph)))

            # print("round graph number edges: ", len(round_graph.edges))
            # if len(round_graph.edges) > 0:
            #    display_graph(round_graph)

            round_subgraphs = [round_graph.subgraph(
                c) for c in nx.connected_components(round_graph)]

            bFound = False
            for subgr in round_subgraphs:
                full_subgr = eras_graph.subgraph(subgr.nodes)
                if len(full_subgr.nodes) < MIN_GRAPH_NODE:
                    continue
                subg_union_fts = union_eras_features(
                    era_ft_dict, full_subgr.nodes)

                # min_w = min(
                #     map(lambda e_d: e_d[2]['weight'], full_subgr.edges.data()))
                # print("min_w: ", min_w)
                # not working -> check improvement of common fts with nodes removal (?)
                # subg_inter_fts = inter_eras_features(
                #     era_ft_dict, full_subgr.nodes)
                # ratio_fts = len(subg_inter_fts) / len(subg_union_fts)
                # print("inter: ", len(subg_inter_fts), " - union: ",
                #      len(subg_union_fts), " - ratio: ", ratio_fts)
                # if ratio_fts < MIN_RATIO_COMMON_FT:
                #    continue

                if (len(full_subgr.nodes) > COND_GRAPH_NODE) or (len(subg_union_fts) > COND_GRAPH_UNION_FTS):
                    bFound = True
                    print("Found subgraph for similarity: ", min_s,
                          " - num ft. union: ", len(subg_union_fts))
                    # display_graph(full_subgr)
                    eras_g_l.append(full_subgr)
                    for n in full_subgr.nodes:
                        del era_ft_dict[str(n)]

            if bFound:
                break

            min_s -= 0.01

    if len(era_ft_dict) >= MIN_GRAPH_NODE:
        remaining_graph = generate_graph_from_dict(era_ft_dict)

        #  REMAINING_SUBGRAPHS
        remaining_subgraphs = [round_graph.subgraph(
            c) for c in nx.connected_components(remaining_graph)]
        remaining_subgraphs = [sg for sg in remaining_subgraphs if len(union_eras_features(
            era_ft_dict, sg.nodes)) > COND_GRAPH_UNION_FTS]

        eras_g_l += remaining_subgraphs

    return eras_g_l


def export_eras_ft_split_file(subsets_dirname, era_ft_dict, e_g, corr_th_str):

    graphs = [e_g.subgraph(c)
              for c in nx.connected_components(e_g)]

    data_subsets = dict({'corr_th': CORR_THRESHOLD})
    data_subsets['original_data_file'] = era_ft_dict['original_data_file']

    data_subsets_eras_ft = {'data_subset_' + str(ind):
                            {'eras': [n for n in g.nodes],
                             'features': union_eras_features(era_ft_dict, g.nodes)}
                            for ind, g in enumerate(graphs)}

    data_subsets['subsets'] = data_subsets_eras_ft

    filename = 'fst_layer_distribution.json'

    try:
        os.makedirs(subsets_dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", subsets_dirname)
            print("Save file at: ", filename)
            with open(filename, 'w') as fp:
                json.dump(data_subsets, fp)
            exit(1)

    filepath = subsets_dirname + '/' + filename
    print("filepath: ", filepath)
    with open(filepath, 'w') as fp:
        json.dump(data_subsets, fp, indent=4)


def main():
    corr_th_str = str(CORR_THRESHOLD).replace('0.', '')
    era_ft_fp = "eras_ft_"+corr_th_str+".json"

    with open(era_ft_fp) as json_file:
        era_ft_dict = json.load(json_file)

    for k, v in list(era_ft_dict.items()):
        if len(v) < MIN_FEATURES:
            del era_ft_dict[k]

    graph_gen_dict = era_ft_dict.copy()
    eras_subg = find_sim_subg(graph_gen_dict)

    full_subg = nx.union_all(eras_subg)
    display_graph(full_subg)

    subsets_dirname = 'data_subsets_' + corr_th_str
    export_eras_ft_split_file(
        subsets_dirname, era_ft_dict, full_subg, corr_th_str)

    # eras_graph.remove_nodes_from(list(nx.isolates(eras_graph)))
    # display_graph(eras_graph)

    # for k in range(30, 2, -1):
    #     print("clique k: ", k)
    #     eras_kclique_g = kclique_graph(eras_graph, k)
    #     if eras_kclique_g is not None:
    #         display_graph(eras_kclique_g)
#
    # eras_subgraphs = [eras_graph.subgraph(c)
    #                   for c in nx.connected_components(eras_graph)]
#
    # for e_s in eras_subgraphs:
    #     print("eras_graph: ", e_s.nodes)
    #     display_graph(e_s)


if __name__ == '__main__':
    main()
