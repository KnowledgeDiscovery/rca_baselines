
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from causalnex.structure.notears import from_pandas
import networkx as nx 

from pyrca.analyzers.ht import HT, HTConfig
from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis, EpsilonDiagnosisConfig
from pyrca.analyzers.rcd import RCD, RCDConfig

import pandas as pd
import networkx as nx
import pickle


def remove_cycles_from_adjacency_matrix(adj_matrix: pd.DataFrame) -> pd.DataFrame:
    G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    while True:
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(*cycle[0][:2])
        except nx.NetworkXNoCycle:
            break
    adj_matrix_no_cycles = nx.to_pandas_adjacency(G, dtype=int)
    adj_matrix_no_cycles = adj_matrix_no_cycles.reindex_like(adj_matrix)
    print("Now, the adjacency matrix does not have cycles.")
    
    return adj_matrix_no_cycles

def data_convert(segment):
    columns = np.array(segment.iloc[:, 1:].columns)
    selector = VarianceThreshold(threshold=0)
    X = segment.iloc[:, 1:].values
    X_var = selector.fit_transform(X)
    idx = selector.get_support(indices=True)
    columns = columns[idx]
    X_var = pd.DataFrame(X_var)
    X_var.columns = list(columns)
    
    return X_var

def rca(ind, segment, model_name):
    segment = segment.iloc[:, 1:]
    print('{} fault starts to detect bayesian structure'.format(ind))
    segment = data_convert(segment)
    columns = np.array(segment.columns)
    #np.save('{}_var_name.npy'.format(ind), columns)  
    X = segment.values
    patch = 100
    sample = X.shape[0]//patch
    X = X[:patch*sample,:]
    X0 = np.sum(X.reshape((-1, patch, X.shape[1])), axis=1)
    X = pd.DataFrame(X0,columns=columns)

    X_train, X_test = train_test_split(X, test_size=0.6, shuffle=False)
    print("Start to run")
    if model_name == "HT":
        model = HT(config=HTConfig(graph=estimated_matrix,root_cause_top_k=10))
        model.train(X_train)
        results = model.find_root_causes(X_test, "label", True).to_list()
    elif model_name == "RCD":
        model = RCD(config=RCDConfig(k=10,alpha_limit=0.5))
        results = model.find_root_causes(X_train,X_test).to_list()
    elif model_name == "ED":
        model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(root_cause_top_k=10))
        model.train(X)
        results = model.find_root_causes(X).to_list()

    print("Saving")
    root_causes  = []
    for result in results:
        root_causes.append([result['root_cause'],result['score']])
    root_causes = pd.DataFrame(root_causes)
    root_causes.columns = [['root_cause','score']]
    root_causes.to_csv("final_{}_{}_root_cause.csv".format(model_name, ind),index=False)

    return


    
with open('../WADI/data_segments.pkl','rb') as f:
    data_segments = pickle.load(f)

models = ['ED', 'RCD', 'HT']
# Run all
for model_name in models:
    for ind,segment in enumerate(data_segments):
        print("Now running {} for data {}.".format(model_name, ind))
        rca(ind, segment,model_name)
        print("-------------------")
