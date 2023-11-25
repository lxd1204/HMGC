close all;  clc
warning off;
addpath(genpath(pwd));
% resultdir = 'Results/';
% if(~exist('Results','file'))
%     mkdir('Results');
%     addpath(genpath('Results/'));
% end

data_path = fullfile(pwd, filesep, "data", filesep);
addpath(data_path);

dirop=dir(fullfile(data_path,'*.mat'));
datasetCandi = {dirop.name}; 
% datasetCandi = {'ORL_Xs'};

runtimes = 2; % run-times on each dataset, default: 1
eta_para = [1e2, 1e3, 1e4];
gamma_para = [10,20,30,40,50];
k_para = [1,2,3,4,5];
% eta_para = [1e4];
% gamma_para = [30,40,50];
% k_para = [1,2,3,4,5];
Iter=5;
normdata=2;
   
for dataIndex = 1: length(datasetCandi)
    dataName = datasetCandi{dataIndex};
    load(dataName);
    disp(dataName);
    fid = fopen([dataName,'_result.txt'],'a');
    
    for eta_Index = 1 : length(eta_para )
        eta_Temp = eta_para(eta_Index);
        for gamma_Index = 1 : length(gamma_para)
            gamma_Temp = gamma_para(gamma_Index);
            for k_Index = 1 : length(k_para)
                k_Temp = k_para(k_Index);
                for rtimes = 1:runtimes
                    result = LSRMSC(data, labels,eta_para(eta_Index),gamma_para(gamma_Index), k_para(k_Index),Iter);
                    %[nmi ACC Purity Fscore Precision Recall AR Entropy]
                    disp(['Dataset: ', datasetCandi{dataIndex}, ...
                        ', --eta_para--: ', num2str(eta_Temp), ', --gamma_para--: ', num2str(gamma_Temp), ...
                        ', --k_para--: ', num2str(k_Temp)]);
                    NMI(rtimes) =result(1);
                    ACC(rtimes) =result(2);
                    Purity(rtimes) = result(3);
                    Fscore(rtimes) = result(4);
                    Precision(rtimes) =result(5);
                    Recall(rtimes) = result(6);
                    ARI(rtimes) = result(7);
                    Entropy(rtimes) =result(8);
                end
                    Result = zeros(10,8);
                    Result(1,:) = [NMI 0 0 0 0 0 0];
                    Result(2,:) = [ACC 0 0 0 0 0 0];
                    Result(3,:) = [Purity 0 0 0 0 0 0];
                    Result(4,:) = [Fscore 0 0 0 0 0 0];
                    Result(5,:) = [Precision 0 0 0 0 0 0];
                    Result(6,:) = [Recall 0 0 0 0 0 0];
                    Result(7,:) = [ARI 0 0 0 0 0 0];
                    Result(8,:) = [Entropy 0 0 0 0 0 0];
                    
                    Result(9,1) = mean(NMI);
                    Result(10,1) = std(NMI);
                    Result(9,2) = mean(ACC);
                    Result(10,2) = std(ACC);
                    Result(9,3) = mean(Purity);
                    Result(10,3) = std(Purity);
                    Result(9,4) = mean(Fscore);
                    Result(10,4) = std(Fscore);
                    Result(9,5) = mean(Precision);
                    Result(10,5) = std(Precision);
                    Result(9,6) = mean(Recall);
                    Result(10,6) = std(Recall);
                    Result(9,7) = mean(ARI);
                    Result(10,7) = std(ARI);
                    Result(9,8) = mean(Entropy);
                    Result(10,8) = std(Entropy);
                   
                    fprintf('Dataset:%s\t eta_par:%f\t gamma_para:%f\t  k_para:%f\t\n',dataName,eta_para(eta_Index),gamma_para(gamma_Index), k_para (k_Index));
                    fprintf(fid,'Dataset:%s\t  eta_par:%f\t gamma_para:%f\t  k_para:%f\t',dataName,eta_para(eta_Index),gamma_para(gamma_Index), k_para (k_Index));
                    fprintf('ACC:%.4f\t nmi:%.4f\t Purity:%4f\t Fscore:%.4f\t Precision:%.4f\t  Recall:%.4f\t AR:%.4f\t Entropy:%.4f\t\n',...
                        Result(9,2),Result(9,1),Result(9,3),Result(9,4),Result(9,5),Result(9,6),Result(9,7),Result(9,8));
                    fprintf(fid,'ACC:%.4f\t nmi:%.4f\t Purity:%.4f\t Fscore:%.4f\t Precision:%.4f\t  Recall:%.4f\t AR:%.4f\t Entropy:%.4f\t\n',...
                        Result(9,2),Result(9,1),Result(9,3),Result(9,4),Result(9,5),Result(9,6),Result(9,7),Result(9,8));
            end
        end
        %         result = LSRMSC(data, labels,eta_para(1), gamma_para(4), k_para(3));
        
    end
%     Result(1,:) = ACC;
%     Result(2,:) =NMI;
%     Result(3,:) = Purity;
%     Result(4,:) = Fscore;
%     Result(5,:) = Precision;
%     Result(6,:) = Recall;
%     Result(7,:) = ARI;
%     Result(8,:) = Entropy;
%     Result(9,1) = mean(ACC);
%     Result(9,2) = std(ACC);
%     Result(10,1) = mean(NMI);
%     Result(10,2) = std(NMI)
%     Result(11,1) = mean(ARI);
%     Result(11,2) = std(ARI);
   
% save([resultdir,char(datasetCandi(dataIndex)),'_result.mat'],'Result');
clear NMI ACC Purity Fscore Precision Recall AR Entropy;
end