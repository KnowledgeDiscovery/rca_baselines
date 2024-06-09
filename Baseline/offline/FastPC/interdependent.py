import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inrc')

    args = parser.parse_args()

    node_file = './0517/output/'+args.input + '_node_all.npy'
    pod_file = './0517/output/'+args.input + '_pod_all.npy'
    mp_file = '/nfs/users/zach/aiops_data/data/0517/p2n.npy'

    node_data = np.load(node_file, allow_pickle=True).item()
    pod_data = np.load(pod_file, allow_pickle=True).item()
    mp_data = np.load(mp_file, allow_pickle=True).item()

    pod_names = pod_data['columns']
    node_names = node_data['columns']

    pod_scores = pod_data['score']
    node_scores = node_data['score']

    p2s = dict(zip(pod_names, pod_scores))
    n2s = dict(zip(node_names, node_scores))

    ctotal = 0
    del_keys = []
    for p in p2s:
        if p not in mp_data:
            del_keys.append(p)
            continue
        node = mp_data[p]
        p2s[p] = p2s[p] * n2s[node]
        ctotal += p2s[p]

    for k in del_keys:
        p2s.pop(k)

    fd = {}
    for p in p2s:
        fd[p] = [p2s[p] / ctotal]
    

    scores = pd.DataFrame.from_dict(fd, orient='index', columns=['ranking_score'])
  
    ranking_score = scores.reset_index(drop=True).to_numpy().reshape(-1)
    ranking_score = preprocessing.normalize([ranking_score]).ravel()
    #print(ranking_score)
    columns = list(scores.index)
    
    #scores = scores.sort_values(by='ranking_score', ascending=False)
    ranking = np.argsort(ranking_score)[::-1]

    K= len(ranking_score)
    #results_combine = {}

    results_combine = pd.DataFrame()
    results_combine['ranking'] = [i+1 for i in range(K)]
    #results_combine = pd.DataFrame(results_combine, columns = ['ranking'])
    results_combine ['pod'] = [columns[ranking[i]] for i in range(K)]
    results_combine ['score'] = [ranking_score[ranking[i]] for i in range(K)]
    results_combine.to_csv('./0517/output/'+ args.input + '_hierarchical_ranking_metrics.csv')
    print(results_combine)
    print('Successfully output the root cause results with considering both node level and pod level')


 
  
    

