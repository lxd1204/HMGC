clc;
clear
close all
addpath(genpath(pwd));
% parameter setting
eta_para = [1e2, 1e3, 1e4];
gamma_para = [10,20,30,40,50];
k_para = [1,2,3,4,5];

% example data
load('Yale_Xs.mat')
Iter=5;
normdata=2;

% =====================  run =====================
result = LSRMSC(data, label,eta_para(1), gamma_para(4), k_para(3));
% result = [nmi ACC Purity Fscore Precision Recall ARI Entropy];
clear data labels