function [label, M,  objHistory,alpha, beta] = HMGC_v1(Ts, nCluster, r, lambda)

[nGraph, nOrder] = size(Ts);
nSmp = size(Ts{1,1}, 1);
alpha = ones(1, nGraph)./nGraph;
beta = ones(nGraph, nOrder)./nOrder;

opt=[];
opt.Dispaly = 'off';
A1s = zeros(nOrder, nOrder, nGraph);
%*********************************************************************
% Merge T and T'
%*********************************************************************
for iGraph = 1:nGraph
    for iOrder = 1:nOrder
        for jOrder = iOrder:nOrder
            e2_ij = sum(sum( Ts{iGraph, iOrder} .* Ts{iGraph, jOrder} ));
            A1s(iOrder, jOrder, iGraph) = e2_ij;
            A1s(jOrder, iOrder, iGraph) = e2_ij;
        end
    end
end
objHistory = [];
maxiter = 20;
myeps = 1e-5;
for iter = 1:maxiter
    %*********************************************************************
    % Merge beta
    %*********************************************************************
    Tbeta = cell(nGraph, 1);
    for iGraph = 1:nGraph
        Ti = zeros(nSmp);
        for iOrder = 1:nOrder
            Ti = Ti + beta(iGraph, iOrder) * Ts{iGraph, iOrder};
        end
        Tbeta{iGraph} = Ti; % nSmp * nSmp
    end

    %*********************************************************************
    % Update W
    %*********************************************************************
    alphaT = zeros(nSmp, nSmp);
    for iGraph = 1:nGraph
        alphaT = alphaT + alpha(iGraph).^r * Tbeta{iGraph};
    end

    val1 = sum(alpha.^r);
    val2 = alphaT;
    M = val2 / val1;
    M = (M + M')/2;
    %     W = solveS(M,nCluster);
    W = zeros(nSmp, nSmp);
    for iSmp = 1:nSmp
        W(iSmp,:) = ProjectOntoSimplex(M(iSmp, :),1);
    end
    obj = compute_obj(Ts, Tbeta,  [], W, alpha, beta, r, lambda, [], []);
    objHistory = [objHistory; obj]; %#ok

    %*********************************************************************
    % Update F
    %*********************************************************************
    %     D = diag(sum(W, 2));
    %     D = (eye(nSmp).^(-0.5));
    %     BB=D;
    %     L = eye(nSmp) - D.^(-0.5).*W.*D.^(-0.5);
    L = eye(nSmp) - W;
    [eigVec, eigVal] = eig1(L, nCluster, 0, 1);
    F = eigVec(:, 1:nCluster);
    obj = compute_obj(Ts, Tbeta, [], W, alpha, beta,r,lambda,L,F);
    objHistory = [objHistory; obj]; %#ok

    %*********************************************************************
    % Update alpha
    %*********************************************************************
    error_alpha = zeros(1,nGraph);
    for iGraph = 1:nGraph
        Ei = W - Tbeta{iGraph};
        error_alpha(iGraph) = sum(sum( Ei.^2 ));
    end
    val3 = (r * error_alpha).^(1/(1-r));
    alpha = val3/sum(val3);
    obj = compute_obj(Ts, Tbeta, error_alpha, W, alpha, beta,r,lambda,L,F);
    objHistory = [objHistory; obj]; %#ok

    %*********************************************************************
    % Update beta
    %*********************************************************************
    for iGraph = 1:nGraph
        fi = zeros(nOrder, 1);
        for iOrder = 1:nOrder
            fi(iOrder) = sum(sum( Ts{iGraph, iOrder} .* W ));
        end
        fi = -2 * alpha(iGraph).^r * fi;

        Hi = 2 * alpha(iGraph).^r * A1s(:, :, iGraph) ;
        max_val = max(max(Hi(:)), max(fi(:)));
        fi = fi/max_val;
        Hi = Hi/max_val;
        Hi = (Hi + Hi')/2;
        [~, o_2] = eig(Hi);
        disp(['min eigval is ', num2str(min(o_2))]);
        lb = zeros(nOrder, 1);
        ub = ones(nOrder, 1);
        %Aeq = ones(1, nOrder);
        %beq = 1;
        [x,~,~] =quadprog(Hi,fi,[],[],[],[],[],[]);
        beta(iGraph, :) = x';
    end
    obj = compute_obj(Ts, Tbeta, error_alpha, W, alpha, beta,r,lambda,L,F);
    objHistory = [objHistory; obj]; %#ok
    if iter > 19 && abs( (objHistory(end-1) - objHistory(end))/objHistory(end-1) ) < myeps
        break;
    end
end

W = (W + W')/2;
d = 1./sqrt(max(sum(W,2),eps));
dW = bsxfun(@times, W, d);
dWd = bsxfun(@times, dW, d');
L = eye(nSmp) - dWd;
L = (L + L')./2;
eigvec = eig1(L, nCluster+1, 0);
eigvec(:, 1) = [];
Y = discretisation(eigvec);
[label, ~] = find(Y');
end


function obj = compute_obj(Ts, Tbeta, error_alpha, W, alpha, beta,r,lambda,L,F)
[nGraph, nOrder] = size(Ts);
nSmp = size(W, 1);
if ~exist('Tbeta', 'var') || isempty(Tbeta)
    Tbeta = cell(nGraph, 1);
    for iGraph = 1:nGraph
        Ti = zeros(nSmp);
        for iOrder = 1:nOrder
            Ti = Ti + beta(iGraph, iOrder) * Ts{iGraph, iOrder};
        end
        Tbeta{iGraph} = Ti; % nSmp * nSmp
    end
end

if ~exist('error_alpha', 'var') || isempty(error_alpha)
    error_alpha = zeros(1,nGraph);
    for iGraph = 1:nGraph
        Ei = W - Tbeta{iGraph};
        error_alpha(iGraph) = sum(sum( Ei.^2 ));
    end
end
LF = L * F;
o2 = sum(sum(F .* LF));
o1 = sum(alpha.^r .* error_alpha);
obj =  o1 + lambda * o2;
end