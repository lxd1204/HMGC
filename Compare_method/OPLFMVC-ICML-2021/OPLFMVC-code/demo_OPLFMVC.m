close all; clear all; clc

addpath(genpath(pwd));
savePath = 'Res/';
if(~exist('Res','file'))
    mkdir('Res');
    addpath(genpath('Res/'));
end
% datasetName = {'WebKB_Xs','100leaves_Xs','20newsgroups','BBCSport_Xs','cornell_2v_Xs','MSRC-v1_Xs','Cora_Xs','wisconsin_2v_Xs',...
%     'texas_2v_Xs','2V_BDGP','mnist4','10X_PBMC_4271n_12206d_8c_uni_Bs',...
%     'prokaryotic','Yale_Xs','ORL_Xs'};
datasetName = {'10X_PBMC_4271n_12206d_8c_uni_Bs'};
for dataIndex = 1: length(datasetName) 
    dataName = [datasetName{dataIndex} '.mat'];
    load(dataName);
    disp(dataName);
    % preprocess
    % load('E:\MVC-AIO\OPLFMVC-ICML-2021\liu21l-supp\OPLFMVC-code\datasets\proteinFold_Kmatrix.mat');
    numclass = length(unique(Y));
    Y(Y==0)=numclass;
    Y=double(Y);
    numker = size(KH,3);
    num = size(KH,1);
    KH = kcenter(KH);
    KH = knorm(KH);
    % algorithm
    for irand = 20
        s=RandStream('mt19937ar','Seed',irand);
        RandStream.setGlobalStream(s);
        tic;
        [Yout, C, WP, Sigma, obj, YB] = onePassLateFusionMVCBeta(KH,numclass);
        fname = fullfile(pwd, [datasetName{dataIndex}, '_tsne.mat'])
        save(fname,'YB');
        [res_mean(:,irand),res_std(:,irand)]= myNMIACCV2(Yout,Y,numclass);
         fname2 = fullfile(pwd, [datasetName{dataIndex}, '_tsne.mat'])
        timecost(irand) = toc;
    end
    % save result
   
    savename = [savePath , char(datasetName(dataIndex)),  '_result.mat'];
    save(savename,'res_mean','res_std','Sigma','timecost' );
    clear res_mean res_std KH Y
end