%
%
%
clear;
clc;
data_path = fullfile(pwd, '.',  filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '.',  filesep, "lib", filesep);
addpath(lib_path);
code_path = genpath(fullfile(pwd, filesep, 'PAPER_DATAUSE'));
addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};
% datasetCandi = {'COIL20_1440n_1024d_20c_uni.mat','CSTR_476n_1000d_4c_tfidf_uni',...
%     'ORL_400n_10304d_40c_uni.mat','tr31_927n_10128d_7c_tfidf_uni.mat',...
%     'Isolet_1560n_617d_26c_uni.mat',...
%     'MSRA25_1799n_256d_12c_uni.mat',...
%     'mfeat_pix_2000n_240d_10c_uni.mat','Zeisel_3005n_4401d_48c_uni.mat',...
%     'MNIST_4000n_784d_10c_uni.mat','Macosko_6418n_8608d_39c_uni.mat'};

res_dir_root = fullfile(pwd,'.','exp_mgc');
pdf_dir_root = fullfile(pwd,  '.', 'exp_params');
exp_n = 'PAPER_DATAUSE';
for i1 = 1 : length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    res_dir = [res_dir_root, filesep, exp_n, filesep, data_name];
    pdf_dir = [pdf_dir_root, filesep, data_name];  
    if ~exist(pdf_dir, 'dir')
        mkdir(pdf_dir);
    end

    
    %*********************************************************************
    % HMGC
    %*********************************************************************
    fname2 = fullfile(res_dir, [data_name, '_HMGC.mat'])
    if exist(fname2, 'file')
       load(fname2);
       [nGrid, nMeasure] = size(HMGC_result);
       HMGC_result(1,130)
       res_acc = HMGC_result(:, 1);
       res_nmi = HMGC_result(:, 2)
       res_ari = HMGC_result(:, 4);
       
       rs = reshape(res_acc, 7, 8);
       bar3(rs);
       xlabel('lambda');
       xticklabels({'10^(-3)', '10^(-2)', '10^(-1)', '10^(0)', '10^(1)', '10^(2)', '10^(3)'});
       ylabel('t');
       yticklabels({'3', '4','5', '6','7','8', '9','10'});
       zlabel('ACC');
       save2pdf(fullfile(pdf_dir, [data_name, '_HMGC_ACC.pdf']), gcf, 600);
       
       rs = reshape(res_nmi, 7, 8);
       bar3(rs);
       xlabel('lambda');
       xticklabels({'10^(-3)', '10^(-2)', '10^(-1)', '10^(0)', '10^(1)', '10^(2)', '10^(3)'});
       ylabel('t');
       yticklabels({'3', '4','5', '6','7','8', '9','10'});
       zlabel('NMI');
       save2pdf(fullfile(pdf_dir, [data_name, '_HMGC_NMI.pdf']), gcf, 600);
       
       rs = reshape(res_ari, 7, 8); 
       bar3(rs);
       xlabel('lambda');
       xticklabels({'10^(-3)', '10^(-2)', '10^(-1)', '10^(0)', '10^(1)', '10^(2)', '10^(3)'});
       ylabel('t');
      yticklabels({'3', '4','5', '6','7','8', '9','10'});
       zlabel('ARI');
       save2pdf(fullfile(pdf_dir, [data_name, '_HMGC_ARI.pdf']), gcf, 600);
    end
end
rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);