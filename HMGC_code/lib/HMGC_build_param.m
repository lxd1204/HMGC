function paramCell = HMGC_build_param(orderCandidate, lambda_candidate)

if ~exist('lambda_candidate', 'var')
    lambda_candidate = [3:10];
end

if ~exist('orderCandidate', 'var')
    orderCandidate = 10.^(-3:3);
end

nParam = length(lambda_candidate) * length(orderCandidate);
paramCell = cell(nParam, 1);
idx = 0;
for i1 = 1:length(orderCandidate)
    for i2 = 1:length(lambda_candidate)
        param = [];
        param.nOrder = orderCandidate(i1);
        param.lambda = lambda_candidate(i2);
        idx = idx + 1;
        paramCell{idx,1} = param;
    end
end
