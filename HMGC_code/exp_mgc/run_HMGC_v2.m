%
%
clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "data1", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
code_path = genpath(fullfile(pwd, '..',  filesep, 'HMGC'));
addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};
% datasetCandi = {'2V_BDGP.mat','10X_PBMC_4271n_12206d_8c_uni_Bs.mat','20newsgroups.mat','100leaves_Xs.mat','BBCSport_Xs.mat',...
%     'mnist4.mat','prokaryotic.mat','texas_2v_Xs.mat','WebKB_Xs.mat',...
%     'wisconsin_2v_Xs.mat','Yale_Xs.mat'};
% datasetCandi = {'3sources.mat'};
datasetCandi = {'SUNRGBD_fea.mat'};
exp_n = 'HMGC';
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
    nCluster = length(unique(Y))-1;
    nSmp = length(Y);

    Xs = cell(1, nView);
    for iView = 1:nView
        Xs{iView} = NormalizeFea(double(X{iView}));
    end


    %*********************************************************************
    % HMGC
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_HMGC.mat']);
    if ~exist(fname2, 'file')
        r = 2;
        knn_size = 10;
        nRepeat = 10;
        nMeasure = 13;
        orderCandidate = [3:10];
        lambda_candidate = 10.^(-3:3);
        paramCell = HMGC_build_param(orderCandidate, lambda_candidate)
        nParam = length(paramCell)
        HMGC_result = zeros(nParam, 1, nRepeat, nMeasure);
        HMGC_time = zeros(nParam, 1);
        for iParam = 1:nParam
            disp(['HMGC iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
            fname3 = fullfile(prefix_mdcs, [data_name, '_HMGC_', num2str(iParam), '.mat']);
            if exist(fname3, 'file')
                load(fname3, 'result_10_s', 'tt');
                HMGC_time(iParam) = tt;
                for iRepeat = 1:nRepeat
                    HMGC_result(iParam, 1, iRepeat, :) = result_10_s(iRepeat, :);
                end
            else
                param = paramCell{iParam};
                nOrder = param.nOrder;
                lambda = param.lambda;
                tic;
                fname4 = fullfile(prefix_mdcs, [data_name, '_HMGC_pre_', num2str(nOrder), '.mat']);
                if exist(fname4, 'file')
                    load(fname4, 'T_CellArray', 't1');
                else
                    tic;
                    T_CellArray = cell(nView, nOrder);
                    S = zeros(nSmp, nSmp, nView);
                    for iView = 1:nView
                        Si = constructW_PKN(Xs{iView}', knn_size, 1);
                        di = sum(Si, 1).^(-.5);
                        Si = (di' .* di) .* Si;
                        Si = (Si + Si')/2;
                        S(:,:,iView) = Si;
                        T_CellArray{iView, 1} = eye(nSmp);
                        T_CellArray{iView, 2} = S(:,:,iView);
                        for iOrder2 = 3:nOrder
                            T_CellArray{iView, iOrder2} = 2*S(:,:,iView)*T_CellArray{iView, iOrder2-1}-T_CellArray{iView, iOrder2-2};
                        end
                    end
                    t1 = toc;
                    save(fname4, 'T_CellArray', 't1');
                end

                tic;
                [H_normalized, label1, W, objHistory] = HMGC_v2(T_CellArray, nCluster, r, knn_size, lambda);
                t2 = toc;
                tt = t1 + t2;
                HMGC_time(iParam) = tt;
                result_10_s = zeros(nRepeat, nMeasure);
                for iRepeat = 1:nRepeat
                    label = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
                    result_10 = my_eval_y(label, Y);
                    HMGC_result(iParam, 1, iRepeat, :) = result_10';
                    result_10_s(iRepeat, :) = result_10';
                end
                save(fname3, 'result_10_s', 'tt', 'param','H_normalized');
            end
        end
        a1 = sum(HMGC_result, 2);
        a3 = sum(a1, 3);
        a4 = reshape(a3, nParam, nMeasure);
        a4 = a4/nRepeat;
        HMGC_result_summary = [max(a4, [], 1), sum(HMGC_time)/nParam];
        save(fname2, 'HMGC_result', 'HMGC_time', 'HMGC_result_summary');

        disp([data_name, ' has been completed!']);
    end
end
rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);