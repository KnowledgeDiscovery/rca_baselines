from __future__ import print_function
from itertools import combinations, permutations
import logging
import networkx as nx
import numpy as np
import scipy.stats as spst
import torch
from numba import cuda
import time
import pickle
import pandas as pd
import miceforest as mf
from sklearn.feature_selection import VarianceThreshold
import argparse
import matplotlib
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)

device = torch.device('cuda')



# This is a function to merge several nodes into one in a Networkx graph
def merge_nodes(G, nodes, new_node):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """
    H = G.copy()

    H.add_node(new_node)

    for n1, n2 in G.edges(data=False):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            H.add_edge(new_node, n2)
        elif n2 in nodes:
            H.add_edge(n1, new_node)

    for n in nodes:
        H.remove_node(n)
    return H


def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.

    Args:
        node_ids: a list of node ids

    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g


def func_z_test(corr_matrix, alpha, ijk, l, g, sep_set, sample_size):
    global cont
    # Move ijk to GPU
    ijk = torch.LongTensor(ijk)
    if cuda:
        ijk = ijk.to(device)

    if l == 0:
        H = corr_matrix[ijk[:, 0:2].repeat(1, 2).view(-1, 2, 2).transpose(1, 2),
                        ijk[:, 0:2].repeat(1, 2).view(-1, 2, 2)]
        if cuda:
            H = H.to(device)
    else:
        M0 = corr_matrix[ijk[:, 0:2].repeat(1, 2).view(-1, 2, 2).transpose(1, 2),
                         ijk[:, 0:2].repeat(1, 2).view(-1, 2, 2)]

        M1 = corr_matrix[ijk[:, 0:2].repeat(1, l).view(-1, l, 2).transpose(1, 2),
                         ijk[:, 2:].repeat(1, 2).view(-1, 2, l)]

        M2 = corr_matrix[ijk[:, 2:].repeat(1, l).view(-1, l, l).transpose(1, 2),
                         ijk[:, 2:].repeat(1, l).view(-1, l, l)]
        if cuda:
            M0 = M0.to(device)
            M1 = M1.to(device)
            M2 = M2.to(device)

        H = M0 - torch.matmul(torch.matmul(M1, torch.inverse(M2)), M1.transpose(2, 1))

    rho_ijs = (H[:, 0, 1] / torch.sqrt(H[:, 0, 0] * H[:, 1, 1]))

    # Absolute value of r, respect cut threshold
    CUT_THR = 0.999999
    rho_ijs = torch.abs(rho_ijs)
    rho_ijs = torch.clamp(rho_ijs, min=0.0, max=CUT_THR)

    #    Note: log1p for more numerical stability, see "Aaux.R";
    z_val = 1 / 2 * torch.log1p((2 * rho_ijs) / (1 - rho_ijs))
    tau = torch.tensor(spst.norm.ppf(1 - alpha / 2) / np.sqrt(sample_size - l - 3) * np.ones(shape=(ijk.shape[0],)),
                       dtype=torch.float32)

    if cuda:
        tau = tau.to(device)

    if cuda:
        ii = ijk[z_val <= tau, 0].cpu().numpy()
        jj = ijk[z_val <= tau, 1].cpu().numpy()
        kk = ijk[z_val <= tau, 2:].cpu().numpy()
    else:
        ii = ijk[z_val <= tau, 0].numpy()
        jj = ijk[z_val <= tau, 1].numpy()
        kk = ijk[z_val <= tau, 2:].numpy()

    for t in range(len(ii)):
        if g.has_edge(ii[t], jj[t]):
            g.remove_edge(ii[t], jj[t])
        cont = True
        sep_set[ii[t]][jj[t]] |= set(kk[t, :])
        sep_set[jj[t]][ii[t]] |= set(kk[t, :])
    return g, sep_set


def estimate_skeleton(corr_matrix, sample_size, alpha, init_graph, know_edge_list, **kwargs):
    global cont
    """Estimate a skeleton graph from the statistis information.

    Args:
        indep_test_func: the function name for a conditional
            independency test.
        data_matrix: data (as a numpy array).
        alpha: the significance level.
        kwargs:
            'max_reach': maximum value of l (see the code).  The
                value depends on the underlying distribution.
            'method': if 'stable' given, use stable-PC algorithm
                (see [Colombo2014]).
            'init_graph': initial structure of skeleton graph
                (as a networkx.Graph). If not specified,
                a complete graph is used.
            other parameters may be passed depending on the
                indep_test_func()s.
    Returns:
        g: a skeleton graph (as a networkx.Graph).
        sep_set: a separation set (as an 2D-array of set()).
    """

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    node_ids = range(corr_matrix.shape[0])

    node_size = corr_matrix.shape[0]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]

    g = init_graph

    l = node_size - 2

    batch_size = 5000

    while l >= 0:
        print(f"==================> Performing round {l} .....")
        cont = False

        ijk = np.empty(shape=(batch_size, (2 + l)), dtype=int)

        index = 0

        for (i, j) in permutations(node_ids, 2):
            ### Known edges
            if know_edge_list:
                if [i, j] in know_edge_list or [j, i] in know_edge_list:
                    continue

            adj_i = list(g.neighbors(i))  # g is actually changed on-the-fly, so we need g_save to test edges
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i, j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
                if len(adj_i) < l:
                    continue
                for k in combinations(adj_i, l):
                    ijk[index, 0:2] = [i, j]  # torch.LongTensor([i, j])  # .cuda(device=device)
                    ijk[index, 2:] = k  # torch.LongTensor(k)       # .cuda(device=device)
                    index += 1
                    if index == batch_size:
                        g, sep_set = func_z_test(corr_matrix, alpha, ijk, l, g, sep_set, sample_size)
                        index = 0

        if index != 0:
            ijk_batch = ijk[:index, :]
            g, sep_set = func_z_test(corr_matrix, alpha, ijk_batch, l, g, sep_set, sample_size)

        l -= 1

    return (g, sep_set)


def estimate_cpdag(skel_graph, sep_set, timeInfoDict, know_edge_list, blacklist_single):
    """Estimate a CPDAG from the skeleton graph and separation sets
    returned by the estimate_skeleton() function.

    Args:
        skel_graph: A skeleton graph (an undirected networkx.Graph).
        sep_set: An 2D-array of separation set.
            The contents look like something like below.
                sep_set[i][j] = set([k, l, m])
        tiers: A dictionary of node lists. {time order: [nodes]}

    Returns:
        An estimated DAG.
    """

    dag = skel_graph.to_directed()
    node_ids = skel_graph.nodes()

    ### Direct based on black list edges
    if blacklist_single:
        for [i, j] in blacklist_single:
            if dag.has_edge(i, j):
                dag.remove_edge(i, j)

    ### Direct based on Known edges
    if know_edge_list:
        for [i, j] in know_edge_list:
            if dag.has_edge(j, i):
                dag.remove_edge(j, i)

    ##### Direct based on Time information
    if timeInfoDict:
        node_time_dict = dict()
        for k, v in timeInfoDict.items():
            for node in v:
                node_time_dict[node] = k

        for (i, j) in combinations(node_ids, 2):
            if i in node_time_dict and j in node_time_dict:
                if node_time_dict[i] > node_time_dict[j] and dag.has_edge(i, j):  # i <---- j
                    _logger.debug('S: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(i, j)
                if node_time_dict[i] < node_time_dict[j] and dag.has_edge(j, i):  # i ----> j
                    _logger.debug('S: remove edge (%s, %s)' % (i, j))
                    dag.remove_edge(j, i)

    ####  V-structure
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        if j in adj_i:
            continue
        adj_j = set(dag.successors(j))
        if i in adj_j:
            continue
        if sep_set[i][j] is None:
            continue
        common_k = adj_i & adj_j
        for k in common_k:
            if k not in sep_set[i][j]:
                if dag.has_edge(k, i):
                    _logger.debug('S: remove edge (%s, %s)' % (k, i))
                    dag.remove_edge(k, i)
                if dag.has_edge(k, j):
                    _logger.debug('S: remove edge (%s, %s)' % (k, j))
                    dag.remove_edge(k, j)

    def _has_both_edges(dag, i, j):
        return dag.has_edge(i, j) and dag.has_edge(j, i)

    def _has_any_edge(dag, i, j):
        return dag.has_edge(i, j) or dag.has_edge(j, i)

    def _has_one_edge(dag, i, j):
        return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
                (not dag.has_edge(i, j)) and dag.has_edge(j, i))

    def _has_no_edge(dag, i, j):
        return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

    #### For all the combination of nodes i and j, apply the following
    #### rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in combinations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    _logger.debug('R2: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            #
            # However, this rule is not necessary when the PC-algorithm
            # is used to estimate a DAG.

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag


def stdmtx(X):
    """
    Convert Normal Distribution to Standard Normal Distribution
    Input:
        X: Each column is a variable
    Output:
        X: Standard Normal Distribution
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    X = X - means[np.newaxis, :]
    X = X / stds[np.newaxis, :]
    return np.nan_to_num(X)


def nameMapping(df):
    ### Map integer to name
    mapping = {i: name for i, name in enumerate(df.columns)}
    return mapping


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plotgraph(g, mapping):
    g = nx.relabel_nodes(g, mapping)
    plt.figure(num=None, figsize=(18, 18), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.shell_layout(g)
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos)

    """
    ### Plot adjacency matrix
    A = nx.adjacency_matrix(g).todense()
    x_labels = mapping.values()
    y_labels = mapping.values()

    print(x_labels)
    print(y_labels)

    fig, ax = plt.subplots()
    im = ax.imshow(A)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, A[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Causal Relations")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    im, cbar = heatmap(A, x_labels, y_labels, ax=ax,
                       cmap="YlGn", cbarlabel="harvest [t/year]")
    texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    fig.tight_layout()
    plt.show()
    """


# graph_excel_single_direction 寻找其中对于目标变量的score即可
def savegraph(gs, corr_matrix, mapping, edgeType, index):
    from collections import Counter, OrderedDict  # if MI_DATASET is 1, still need to run this
    edges_all = [e for g in gs for e in list(g.edges)]
    edges_appear_count = Counter(edges_all)
    edges_keep = edges_all  # [v  for v, num in edges_all.items() if num == MI_DATASET]
    g = nx.empty_graph(corr_matrix.shape[0], create_using=nx.DiGraph())
    g.add_edges_from(edges_keep)

    ### save edges to excel
    strength = []
    g_edges = list(g.edges)  # all edges

    for (i, j) in g_edges:
        if edgeType == 's':
            if cuda:
                # print(corr_matrix[i, j].cpu().item())
                strength.append(corr_matrix[i, j].cpu().item())
            else:
                # print(corr_matrix[i, j].item())
                strength.append(corr_matrix[i, j].item())
        elif edgeType == 'c':
            strength.append(edges_appear_count[(i, j)])

        else:
            strength.append(np.nan)

    graph_excel = {'Cause': [mapping[e[0]] for e in g_edges], 'Effect': [mapping[e[1]] for e in g_edges],
                   'Strength': [round(a, 3) for a in strength]}
    graph_excel = pd.DataFrame.from_dict(graph_excel)
    graph_excel = graph_excel[graph_excel['Effect'] == 'label']
    graph_excel.to_csv("{}_graph_excel.csv".format(index), index=False)

    ### Seperate Single and Bidirectional edges
    # graph_excel_single = {'Cause': [], 'Effect': [], 'Strength':[]}
    # graph_excel_bi = {'Cause': [], 'Effect': [], 'Strength':[]}
    # for m, (i, j) in enumerate(g_edges):
    #     if (j, i) not in g_edges: # Single directional
    #         graph_excel_single['Cause'].append(mapping[i])
    #         graph_excel_single['Effect'].append(mapping[j])
    #         graph_excel_single['Strength'].append(strength[m])
    #     else:  # bidirectional edges
    #         graph_excel_bi['Cause'].append(mapping[i])
    #         graph_excel_bi['Effect'].append(mapping[j])
    #         graph_excel_bi['Strength'].append(strength[m])
    #
    # graph_excel_single = pd.DataFrame.from_dict(graph_excel_single)
    # graph_excel_single.to_csv("{}_graph_excel_single_direction.csv".format(index), index=False)
    # graph_excel_bi = pd.DataFrame.from_dict(graph_excel_bi)
    # graph_excel_bi.to_csv("{}_graph_excel_bidirection.csv".format(index), index=False)


def getblackList(df, blacklist, node_size):
    node_ids = range(node_size)
    init_graph = _create_complete_graph(node_ids)
    black_edges = set()
    with open(blacklist, 'rb') as f:
        for line in f.readlines():
            cause, effect = line.splitlines()[0].decode("utf-8").split(',')
            i, j = df.columns.get_loc(cause.strip()), df.columns.get_loc(effect.strip())
            black_edges |= {(i, j)}

    blacklist_single = set((i, j) for (i, j) in black_edges if (j, i) not in black_edges)
    init_graph.remove_edges_from([(i, j) for (i, j) in black_edges if i < j and (j, i) in black_edges])

    return init_graph, blacklist_single


def getTiers(tiers, mapping_r):
    with open(tiers, 'rb') as f:
        timeinfodict = dict()
        n = 1
        for line in f.readlines():
            line = line.splitlines()[0].decode("utf-8").split(',')
            line = [mapping_r[i.strip()] for i in line]
            timeinfodict[n] = line
            n += 1
    return timeinfodict


def getknownedges(knownedges, mapping_r):
    #     nonlocal  mapping_r
    know_edge_list = []
    with open(knownedges, 'rb') as f:
        for line in f.readlines():
            cause, effect = line.splitlines()[0].decode("utf-8").split(',')
            know_edge_list.append([mapping_r[cause.strip()], mapping_r[effect.strip()]])
    return know_edge_list


def Fast_PC_Causal_Graph(df, alpha, cuda):
    ## check corr=1
    corr = np.corrcoef(df.values.T)

    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr[i, j]) > 0.999999:
                raise Exception('Feature ' + str(df.columns[i]) + ' and feature ' + str(
                    df.columns[j]) + ' are strongly correlated ' + str(
                    corr[i, j]) + ', you might want to delete one feature.')

    mapping = {i: name for i, name in enumerate(df.columns)}
    mapping_r = {name: i for i, name in mapping.items()}

    if df.isnull().values.any():
        imputation = True

    ### Multiple Imputation
    datasets = []

    datasets.append(df)

    gs = []
    for df in datasets:
        N = df.shape[0]
        node_size = df.shape[1]

        corr_matrix = np.corrcoef(df.values.T)
        corr_matrix = torch.tensor(corr_matrix, dtype=torch.float32)

        # stablize
        corr_matrix += 1e-6 * np.random.random(corr_matrix.shape)

        if cuda:
            corr_matrix = corr_matrix.to(device)

        st = time.time()

        ### Blacklist

        init_graph = _create_complete_graph(range(node_size))
        blacklist_single = None
        timeInfoDict = None
        know_edge_list = []

        (g, sep_set) = estimate_skeleton(corr_matrix=corr_matrix,
                                         sample_size=N,
                                         alpha=alpha,
                                         init_graph=init_graph,
                                         know_edge_list=know_edge_list,
                                         method='stable')

        g = estimate_cpdag(skel_graph=g,
                           sep_set=sep_set,
                           timeInfoDict=timeInfoDict,
                           know_edge_list=know_edge_list,
                           blacklist_single=blacklist_single)

        en = time.time()
        print("Total running time:", en - st)
        print('Edges are:', g.edges(), end='')

        ### Integer to real name
        gs.append(g)
        # plotgraph(g, mapping)
    matrix = nx.to_numpy_matrix(gs[0])
    return matrix