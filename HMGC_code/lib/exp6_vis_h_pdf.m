%
%
clear;
clc;
pwd
data_path = fullfile(pwd, '.',  filesep, "data1", filesep)
addpath(data_path);
lib_path = fullfile(pwd, '.',  filesep, "lib", filesep);
addpath(lib_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};


exp_n = 'HMGCL4';
% profile off;
% profile on;
for i1 = 1 : length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    prefix_mdcs = [pwd,filesep, exp_n, filesep, data_name];
    
    load(data_name);
    if exist('y', 'var')
        Y = y;
    end
    if size(X, 1) ~= size(Y, 1)
        Y = Y';
    end
%     assert(size(X, 1) == size(Y, 1));
    nSmp = size(X, 1);
    nCluster = length(unique(Y));
    
    
    %*********************************************************************
    % MKCSS-TNNLS-2020
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_HMGC1.mat']);
    if exist(fname2, 'file')
        load(fname2);
        H = H_normalized;
        rng(0);  % 设置随机种子以确保结果可复现
        as = tsne(H);
        gscatter(as(:,1), as(:,2), Y);
        legend('off');
        axis off;
        clear H;
        fname3 = fullfile(prefix_mdcs, [data_name, '_HMGC.pdf']);
        save2pdf(fname3, gcf, 600);
    end
    
end
rmpath(data_path);
rmpath(lib_path);

