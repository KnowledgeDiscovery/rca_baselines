import numpy as np
import pandas as pd
import pickle
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import VarianceThreshold
from baro_algorithm import bocpd, robust_scorer

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
    
with open('../WADI/data_segments.pkl','rb') as f:
    data_segments = pickle.load(f)


for ind,segment in enumerate(data_segments):
    segment = segment.iloc[:, 1:]
    print('{} fault starts to detect bayesian structure'.format(ind))
    segment = data_convert(segment)
    columns = np.array(segment.columns)
    #np.save('{}_var_name.npy'.format(ind), columns)  
    X = segment.values
    patch = 100
    sample = X.shape[0]//patch
    X = X[:patch*sample,:]
    X = np.sum(X.reshape((-1, patch, X.shape[1])), axis=1)
    X_df = pd.DataFrame(X,columns=columns)
    anomalies = bocpd(X_df)
    print("Anomalies are detected at timestep:", anomalies[0])
    results = robust_scorer(X_df,anomalies=anomalies)

    root_causes  = []
    for result in results:
        (root_cause, score) = result
        root_causes.append([root_cause, score])
    root_causes = pd.DataFrame(root_causes)
    root_causes.columns = [['root_cause','score']]
    root_causes.to_csv("./final_{}_root_cause.csv".format(ind),index=False)







