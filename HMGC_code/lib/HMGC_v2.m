function [H_normalized, label, M, objHistory,alpha, beta] = HMGC_v2(Ts, nCluster, r, k, lambda)

if ~exist('r', 'var') || isempty(r)
    r = 2;
end

if ~exist('k', 'var') || isempty(k)
    k = 10;
end

if ~exist('islocal', 'var') || isempty(islocal)
    islocal = 1;
end

[nGraph, nOrder] = size(Ts);
nSmp = size(Ts{1,1}, 1);
alpha = ones(1, nGraph)./nGraph;
beta = ones(nGraph, nOrder)./nOrder;

opt = [];
opt.Display = 'off';
A1s = zeros(nOrder, nOrder, nGraph);
%*********************************************************************
% Merge T and T'
%*********************************************************************
T0 = zeros(nSmp);
for iGraph = 1:nGraph
    T0 = T0 + Ts{iGraph, 2};
    for iOrder = 1:nOrder
        for jOrder = iOrder:nOrder
            e2_ij = sum(sum( Ts{iGraph, iOrder} .* Ts{iGraph, jOrder} ));
            A1s(iOrder, jOrder, iGraph) = e2_ij;
            A1s(jOrder, iOrder, iGraph) = e2_ij;
        end
    end
end

T0 = (T0 + T0')/2;
D = diag(sum(T0));
L = D - T0;
[F, ~, ev] = eig1(L, nCluster, 0);
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end


objHistory = [];
maxiter = 50;
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
    distf = L2_distance_1(F',F');
    val1 = sum(alpha.^r);
    M = alphaT - .5 * lambda * distf;
    M = M / val1;
    M = (M + M')/2;
    [~, idx] = sort(M, 2, 'descend');
    W = zeros(nSmp, nSmp);
    for iSmp = 1:nSmp
        if islocal == 1
            idxa0 = idx(iSmp, 2: k + 1);
        else
            idxa0 = 1:nSmp;
        end
        W(iSmp, idxa0) = ProjectOntoSimplex(M(iSmp, idxa0), 1);
    end
    %     obj = compute_obj(Ts, Tbeta,  [], W, alpha, beta, r, lambda, [], []);
    %     objHistory = [objHistory; obj]; %#ok

    %*********************************************************************
    % Update F
    %*********************************************************************
    W = (W + W')/2;
    D = diag(sum(W));
    L = D - W;
    %     F_old = F;
    [F, ~, ev] = eig1(L, nCluster, 0);
    %     fn1 = sum(ev(1:nCluster));
    %     fn2 = sum(ev(1:nCluster+1));
    %     if fn1 > 0.00000000001
    %         lambda = 2*lambda;
    %     elseif fn2 < 0.00000000001
    %         lambda = lambda/2;
    %         F = F_old;
    %     else
    %         break;
    %     end
%     obj = compute_obj(Ts, Tbeta, [], W, alpha, beta,r,lambda,L,F);
%     objHistory = [objHistory; obj]; %#ok

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
    %     obj = compute_obj(Ts, Tbeta, error_alpha, W, alpha, beta,r,lambda,L,F);
    %     objHistory = [objHistory; obj]; %#ok

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
        % [~, o_2] = eig(Hi);
        % disp(['min eigval is ', num2str(min(diag(o_2)))]);
        lb = zeros(nOrder, 1);
        ub = ones(nOrder, 1);
        %Aeq = ones(1, nOrder);
        %beq = 1;
        [x,~,~] = quadprog(Hi,fi,[],[],[],[],[],[],[],opt);
        beta(iGraph, :) = x'/nOrder;
    end
    obj = compute_obj(Ts, Tbeta, error_alpha, W, alpha, beta,r,lambda,L,F);
    objHistory = [objHistory; obj]; %#ok
    if iter > 5 && abs( (objHistory(end-1) - objHistory(end))/objHistory(end-1) ) < myeps
        break;
    end
end
Zt = ( abs(W) + abs(W') ) / 2 ;
CKSym = BuildAdjacency(thrC(Zt, 0.7));
H_normalized = SpectralClustering_ncut(CKSym, nCluster);
label = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
% W = (W + W')/2;
% d = 1./sqrt(max(sum(W,2),eps));
% dW = bsxfun(@times, W, d);
% dWd = bsxfun(@times, dW, d');
% L = eye(nSmp) - dWd;
% L = (L + L')./2;
% eigvec = eig1(L, nCluster+1, 0);
% eigvec(:, 1) = [];
% Y = discretisation(eigvec);
% [label, ~] = find(Y');
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