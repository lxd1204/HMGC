import numpy as np
from W_Construct import norm_W,KNN
from data_loader import load_mat
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import adjusted_rand_score
import argparse
import logging
import warnings
import utils

import os
import glob
import scipy.io
import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore",category=DeprecationWarning)
parser = argparse.ArgumentParser(description='AONGR')
parser.add_argument('--epochs', '-te', type=int, default=30, help='number of interation')
parser.add_argument('--k', type=int, default=20, help='the number of neighbors, COIL20:2, Yale:2, HW:10, Wikipedia:35, Caltech101:20')
parser.add_argument('--index', type=float, default=0, help='view number,COIL20:1, Yale:0, HW:1, Wikipedia:1,Caltech101:3')
# parser.add_argument('--dataset', type=str, default='MSRCV1', help='choose a dataset')

args = parser.parse_args()
def AONGR(X,n_knn,n_cluster,n_iter,view_index,lambda_1,lambda_2):
    n_view = X.size
    N,_ = X[0].shape
    W = []
    for i in range(n_view):
        W.append(norm_W(KNN(X[i],n_knn)))
    alpha = np.ones(n_view)/n_view
    W_new = np.zeros((N, N))
    for i in range(n_view):
        W_new += W[i] * alpha[i]
    val, vec = np.linalg.eigh(W[view_index])
    F = vec[:, -n_cluster:]
    G,obj,FG,F1,alpha1= opt(W,F,alpha,n_iter,n_cluster,lambda_1=lambda_1,lambda_2=lambda_2)
    y_pred = np.argmax(G,axis=1)+1
    return y_pred,obj,G,W_new,FG,F1,alpha1

def opt(W,  F, alpha, NITER,n_cluster,lambda_1=1,lambda_2=1):

    n_view = alpha.size
    N, _ = W[0].shape
    b = np.zeros(n_view)
    M = np.zeros((n_view,n_view))
    G = F
    for i in range(n_view):
        for j in range(n_view):
            M[i][j] = np.trace(W[i]@W[j].T)
    if np.max(M)>1e2:
        M = norm_W(M)

    M = M + np.identity(n_view) *lambda_2
    OBJ = []
    for Iter in range(NITER):
        # 更新alpha
        W_new = np.zeros((N, N))
        for i in range(n_view):
            b[i] = np.trace(W[i].T @ (F @ G.T))
        alpha = SimplexQP_ALM(M, 2*b, alpha)
        for i in range(n_view):
            W_new += W[i]*alpha[i]
        # update G

        G = (W_new @ F + lambda_1 * F) / (1 + lambda_1)
        G[G<0]=0
        #update F
        Q = (W_new@G+lambda_1*G)
        U, lambda_s, VT = np.linalg.svd(Q)
        A = np.concatenate((np.identity(n_cluster), np.zeros((N - n_cluster, n_cluster))), axis=0)
        F = U @ A @ VT

        obj = np.square(np.linalg.norm(W_new-F@G.T))+\
              lambda_1*np.square(np.linalg.norm(F-G))+lambda_2*alpha.T@alpha
        OBJ.append(obj)

    return G,OBJ,F@G.T,F,alpha

def EProjSimplex_new(v,k=1):
    ft = 1
    n = v.size
    v_0 = v - np.mean(v) + k/n
    v_min = np.min(v_0)
    if v_min<0:
        f = 1
        lambda_m = 0
        while abs(f)> 1e-10:
            v_1 = v_0 - lambda_m
            posidx = v_1>0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v_1[posidx])-k
            lambda_m = lambda_m-f/g
            ft = ft+1
            if ft>100:
                v_1[v_1<0]=0
                x = v_1
                break
            v_1[v_1<0]=0
            x = v_1
    else:
        x = v_0
    return x


def SimplexQP_ALM(A,b,x,mu=5,beta=1.5):
    N_inter = 500
    threshold = 1e-8
    val = 0
    v = np.ones(x.size)
    lambda_n = np.ones(x.size)
    cnt = 0

    for i in range(N_inter):
        x = EProjSimplex_new(v-1/mu*(lambda_n+A@v-b))
        v = x+1/mu*(lambda_n-A.T@x)
        lambda_n = lambda_n + mu*(x-v)
        mu = beta*mu
        val_old = val
        val = x.T@A@x -x.T@b

        if abs(val - val_old) < threshold:
            if cnt >=5:
                break
            else:
                cnt +=1
        else:
            cnt = 0

    return x

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)


def classification_metric(y_true, y_pred, average='macro', verbose=False, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    if verbose:
        # print('Confusion Matrix')
        # print(confusion_matrix)
        logging.info('accuracy: {}, precision: {}, recall: {}, f_measure: {}'.format(accuracy, precision, recall, f_score))
    #print({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix)
    return f_score,recall,precision


if __name__=="__main__":
    # 指定文件夹路径
    folder_path = r'G:\method\AONGR-IS-2023\AONGR-main\data'
    # 构建匹配模式，查找所有的.mat文件
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))
    # 遍历找到的.mat文件

    for matfile in mat_files:
        print(f'Processed file: {matfile}')
        last_part = os.path.splitext(os.path.basename(matfile))[0]
        file_path = r'G:\method\AONGR-IS-2023\AONGR-main\{}.txt'.format(last_part)
        x = print(file_path)
        X,GT = load_mat('{}'.format(matfile))
        # X,GT = load_mat('data/{}.mat'.format(args.dataset))
        GT = GT.reshape(np.max(GT.shape), )
        n_cluster = len(np.unique(GT))
        lambda_1 = [1e-3, 1e-2, 1e-1, 1, 10, 1e2]
        lambda_2 = [5, 10, 20, 40, 60, 80]
        ACC_total = []
        for i in range(6):
            for j in range(6):
                y_pred,obj,G,W,FG,F,alpha= AONGR(X,args.k,n_cluster,args.epochs,args.index,lambda_1=lambda_1[i],lambda_2=lambda_2[j])
                ACC = acc(GT,y_pred)
                NMI = metrics.normalized_mutual_info_score(GT,y_pred)
                Purity = purity_score(GT,y_pred)
                ARI = adjusted_rand_score(GT,y_pred)
                Fscore,Recall,Precision =classification_metric(GT,y_pred)
                #from sklearn.metrics import confusion_matrix
                from sklearn.metrics import classification_report
                from sklearn.metrics import cohen_kappa_score



                # s = classification_report(GT,y_pred, output_dict=True)['weighted avg']
                # Precision = s['precision']
                # Recall = s['recall']
                # Fscore = s['f1-score']

                #Precision = precision_score(GT,y_pred, average='macro')
                #Recall = recall_score(GT,y_pred, average='macro')
                #Fscore = f1_score(GT,y_pred, average='macro')

                #Fscore,Precision,Recall = compute_f(GT,y_pred)

                print('clustering accuracy: {}, NMI: {}, Purity: {},ARI: {},FScore: {}, Precision: {}, Recall: {}lambda_1:{},lambda_2:{}'.format(ACC,NMI,Purity,
                                                                                                   ARI ,Fscore,Precision,Recall,lambda_1[i],lambda_2[j]))
                output_result_str = ('clustering accuracy: {}, NMI: {}, Purity: {},ARI: {},FScore: {}, Precision: {}, Recall: {}lambda_1:{},lambda_2:{}\n'.format(ACC,NMI,Purity,
                                                                                                   ARI ,Fscore,Precision,Recall,lambda_1[i],lambda_2[j]))
                with open(file_path, 'a') as file:
                    file.write(output_result_str)








