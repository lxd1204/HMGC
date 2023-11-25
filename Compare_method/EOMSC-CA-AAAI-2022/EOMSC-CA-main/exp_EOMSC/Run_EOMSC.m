clear
clc
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd,'..',filesep,"lib",filesep);
addpath(lib_path);
code_path =  genpath(fullfile(pwd,'..',filesep,'EOKSC'));
addpath(code_path);

dirop=dir(fullfile(data_path,'*.mat'));
datasetCandi = {dirop.name}; %已经包含data目录下的所有mat数据集

endmetric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};
anchor_rate=[1 2 3 4 5 6 7];
d_rate = [1 2 3 4 5 6 7];

exp_n = 'EOMSC';
for i1 = 1 : length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name,'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir:',dir_name,'failed,check the authorization']);
    end
    clear X  Y;
    load(data_name);
    res = zeros(1, 8);
    
    sum = zeros(1,8);  % 为了求多从结果的平均值
    
    if exist('y', 'var')
        Y = y;
    end
    Y = Y(:);
    
    %*********************************************************************
    % EOMSC
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs,[data_name,'_EOKSC.mat']);
    if ~exist(fname2,'file')
        k = length(unique(Y));
        n = length(Y);
        for ichor = 1:length(anchor_rate)
            for id = 1:length(d_rate)
                fname3 = fullfile(prefix_mdcs,[data_name,  '-ACC=',num2str(res(1, 1)), '-anchor_rate=',...
                    num2str(ichor), '-d_rate=', num2str(id),'_EOKSC.mat']);
                if exist(fname3,'file')
                    load(fname3, 'res');
                else
                    tic;
                    [A,W,Z,iter,obj,alpha,label] = algo_qp(X,Y,d_rate(id)*k,anchor_rate(ichor)*k);
                    res = Clustering8Measure(Y, label);
                    res
                    sum = sum + res;
                    timer(ichor,id)  = toc;
                    resall{ichor,id} = res;
                    objall{ichor,id} = obj;
                    save(fullfile(prefix_mdcs,[data_name,  '-ACC=',num2str(res(1, 1)), '-anchor_rate=',...
                    num2str(ichor), '-d_rate=', num2str(id),'_EOKSC.mat']), 'res');
                end
            end
        end
        final_result = sum/(length(anchor_rate)*length(dir_name));
        save(fname2, 'final_result');
    end
end