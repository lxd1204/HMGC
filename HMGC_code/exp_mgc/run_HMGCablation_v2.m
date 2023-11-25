%
%
%

clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
code_path = genpath(fullfile(pwd, '..',  filesep, 'HMGC'));
addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

datasetCandi = {'WebKB_Xs.mat','3sources.mat','COIL20.mat','cornell_2v_Xs.mat','Cora_Xs.mat','texas_2v_Xs.mat',...
    'SUNRGBD_fea.mat','small_NUS.mat','scene-15.mat','mnist4.mat','100leaves_Xs.mat','20newsgroups.mat',...
    '10X_PBMC_4271n_12206d_8c_uni_Bs.mat','2V_BDGP.mat',};
exp_n = 'HMGCablation';
for i1 = 1 : length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name, 'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir: ',dir_name, 'failed, check the authorization']);
    end

    clear X y Y;
    load(data_name);
    if exist('y', 'var')
        Y = y;
    end
    Y = Y(:);

    nView = length(X);
    nCluster = length(unique(Y));
    nSmp = length(Y);

    Xs = cell(1, nView);
    for iView = 1:nView
        Xs{iView} = NormalizeFea(double(X{iView}));
    end


    %*********************************************************************
    % HMGCablation
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_HMGCablation.mat']);
    if ~exist(fname2, 'file')
        r = 2;
        knn_size = 10;
        nRepeat = 2;
        nMeasure = 13;
        lambda_candidate = 10.^(-3:3);
        paramCell = HMGCablation_build_param(lambda_candidate);
        nParam = length(paramCell);
        HMGCablation_result = zeros(nParam, 1, nRepeat, nMeasure);
        HMGCablation_time = zeros(nParam, 1);
        for iParam = 1:nParam
            disp(['HMGCablation iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
            fname3 = fullfile(prefix_mdcs, [data_name, '_HMGCablation_', num2str(iParam), '.mat']);
            if exist(fname3, 'file')
                load(fname3, 'result_10_s', 'tt');
                HMGCablation_time(iParam) = tt;
                for iRepeat = 1:nRepeat
                    HMGCablation_result(iParam, 1, iRepeat, :) = result_10_s(iRepeat, :);
                end
            else
                param = paramCell{iParam};
                lambda = param.lambda;
                tic;
                fname4 = fullfile(prefix_mdcs, [data_name, '_HMGCablation_pre.mat']);
                if exist(fname4, 'file')
                    load(fname4, 'T_CellArray', 't1');
                else
                    tic;
                    T_CellArray = cell(nView, 1);
                    for iView = 1:nView
                        Si = constructW_PKN(Xs{iView}', knn_size, 1);
                        di = sum(Si, 1).^(-.5);
                        Si = (di' .* di) .* Si;
                        Si = (Si + Si')/2;
                        Si = sparse(Si);
                        T_CellArray{iView, 1} = Si;
                    end
                    t1 = toc;
                    save(fname4, 'T_CellArray', 't1');
                end

                tic;
                [H_normalized, label1, W, objHistory] = HMGC_ablation(T_CellArray, nCluster, r, knn_size, lambda);
                t2 = toc;
                tt = t1 + t2;
                HMGCablation_time(iParam) = tt;
                result_10_s = zeros(nRepeat, nMeasure);
                for iRepeat = 1:nRepeat
                    label = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
                    result_10 = my_eval_y(label, Y);
                    HMGCablation_result(iParam, 1, iRepeat, :) = result_10';
                    result_10_s(iRepeat, :) = result_10';
                end
                save(fname3, 'result_10_s', 'tt', 'param');
            end
        end
        a1 = sum(HMGCablation_result, 2);
        a3 = sum(a1, 3);
        a4 = reshape(a3, nParam, nMeasure);
        a4 = a4/nRepeat;
        HMGCablation_result_summary = [max(a4, [], 1), sum(HMGCablation_time)/nParam];
        save(fname2, 'HMGCablation_result', 'HMGCablation_time', 'HMGCablation_result_summary');

        disp([data_name, ' has been completed!']);
    end
end
rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);