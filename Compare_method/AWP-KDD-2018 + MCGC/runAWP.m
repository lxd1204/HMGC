close all;  clc
warning off;
addpath(genpath(pwd));

data_path = fullfile(pwd,filesep, "data", filesep);
addpath(data_path);
code_path = fullfile(pwd, filesep, "AWP", filesep);
addpath(code_path);

dirop=dir(fullfile(data_path,'*.mat'));
datasetCandi = {dirop.name}; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For text dataset (features are word frequence),
% use cosine metric

for dataIndex = 1: length(datasetCandi)
    dataName = datasetCandi{dataIndex};
    load(dataName);
    disp(dataName);
    result_path = fullfile(code_path,[dataName,'_result.txt']);
    fid = fopen(result_path,'a');
    numClust = length(unique(gt));
    knn0 = 20;
    metric = 'squaredeuclidean';
    
    % load('./data/BBCSport.mat')
    % numClust = length(unique(gt));
    % knn0 = 10;
    % metric = 'cosine';
    [label] = AWP_main(fea, numClust, knn0, metric);
    score = getFourMetrics(label, gt);
%    score = [ACC nmi Purity Fscore Precision Recall AR Entropy];
    
    fprintf('Dataset:%s\t ACC:%.4f\t nmi:%.4f\t Purity:%4f\t Fscore:%.4f\t Precision:%.4f\t  Recall:%.4f\t AR:%.4f\t \n',dataName,...
        score(1),score(2),score(3),score(4),score(5),score(6),score(7));
    fprintf(fid,'Dataset:%s\t ACC:%.4f\t nmi:%.4f\t Purity:%4f\t Fscore:%.4f\t Precision:%.4f\t  Recall:%.4f\t AR:%.4f\t\n',dataName,...
        score(1),score(2),score(3),score(4),score(5),score(6),score(7));
end