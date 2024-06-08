import pandas as pd

folders = ['0606','0517','0524','0901','1203']
predicts = []
for fd in folders:
    pods_data = pd.read_csv(fd+'/output/Pod_level_combine_ranking.csv')
    pods = list(pods_data['pod'])
    # pods = [x.split('_')[1] for x in pods]
    predicts.append(pods)

k = [1,3,5,7,10]

def precision_on_topk(predicts,reals,k):
    pr = 0
    for pred, real in zip(predicts, reals):
        pred = pred[:k]
        hit_count = len(set(pred) & set(real))
        min_len = min(k,len(real))
        pr += hit_count/min_len
    return pr/len(reals)

def mean_precision_k(predicts,reals,k):
    pr = 0
    for i in range(1,k+1):
        pr += precision_on_topk(predicts,reals,i)
    return pr/k

def mrr(predicts,reals):
    mrr_val = 0
    for preds,real in zip(predicts,reals):
        tmp = []
        for real_item in real:
            index = preds.index(real_item) if real_item in preds else sys.maxsize-1
            tmp.append(index+1)
        mrr_val += 1/min(tmp)
    return mrr_val/len(reals)

reals = [['productpage-v1-5f9dbcd669-z2prs'],
         ['catalogue-8667bb6cbc-hqzfw'],
         ['catalogue-85fd4965b7-q8477'],
         ['catalogue-6c7b9b975-xfjps'],
         ['mongodb-v1-64c6b69879-p4wfp']]

for item in k:
    pr = precision_on_topk(predicts,reals,item)
    map_val = mean_precision_k(predicts,reals,item)
    mrr_val = mrr(predicts,reals)
    print("pr@{}:{} map@{}:{} mrr:{}".format(item,pr,item,map_val,mrr_val))
