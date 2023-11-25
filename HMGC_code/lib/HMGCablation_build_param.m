function paramCell = HMGCablation_build_param(lambda_candidate)

if ~exist('lambda_candidate', 'var')
    lambda_candidate = [3:10];
end

nParam = length(lambda_candidate);
paramCell = cell(nParam, 1);
idx = 0;
for i1 = 1:length(lambda_candidate)
    param = [];
    param.lambda = lambda_candidate(i1);
    idx = idx + 1;
    paramCell{idx,1} = param;
end
end
