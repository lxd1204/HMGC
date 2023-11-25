function [H_normalized, label, M, objHistory,alpha, beta] = HMGC_ablation(Ts, nCluster, r, k, lambda)

if ~exist('r', 'var') || isempty(r)
    r = 2;
end

if ~exist('k', 'var') || isempty(k)
    k = 10;
end
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end
if ~exist('islocal', 'var') || isempty(islocal)
    islocal = 1;
end

nGraph = size(Ts, 1);
nSmp = size(Ts{1,1}, 1);
alpha = ones(1, nGraph)./nGraph;

opt = [];
opt.Display = 'off';
T0 = zeros(nSmp);
for iGraph = 1:nGraph
    T0 = T0 + Ts{iGraph, 1};
end
T0 = (T0 + T0')/2;
D = diag(sum(T0));
L = D - T0;
[F, ~, ev] = eig1(L, nCluster, 0);


objHistory = [];
maxiter = 50;
myeps = 1e-5;
for iter = 1:maxiter

    %*********************************************************************
    % Update W
    %*********************************************************************
    alphaT = zeros(nSmp, nSmp);
    for iGraph = 1:nGraph
        alphaT = alphaT + alpha(iGraph).^r * Ts{iGraph};
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
        Ei = W - Ts{iGraph};
        error_alpha(iGraph) = sum(sum( Ei.^2 ));
    end
    val3 = (r * error_alpha).^(1/(1-r));
    alpha = val3/sum(val3);
    %     obj = compute_obj(Ts, Tbeta, error_alpha, W, alpha, beta,r,lambda,L,F);
    %     objHistory = [objHistory; obj]; %#ok

    obj = compute_obj(Ts, error_alpha, W, alpha, r,lambda,L,F);
    objHistory = [objHistory; obj]; %#ok
    if iter > 5 && abs( (objHistory(end-1) - objHistory(end))/objHistory(end-1) ) < myeps
        break;
    end
end
Zt = ( abs(W) + abs(W') ) / 2 ;
CKSym = BuildAdjacency(thrC(Zt, 0.7));
H_normalized = SpectralClustering_ncut(CKSym, nCluster);
label = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
end


function obj = compute_obj(Ts, error_alpha, W, alpha, r,lambda,L,F)
nGraph = size(Ts,1);
if ~exist('error_alpha', 'var') || isempty(error_alpha)
    error_alpha = zeros(1,nGraph);
    for iGraph = 1:nGraph
        Ei = W - Ts{iGraph};
        error_alpha(iGraph) = sum(sum( Ei.^2 ));
    end
end
LF = L * F;
o2 = sum(sum(F .* LF));
o1 = sum(alpha.^r .* error_alpha);
obj =  o1 + lambda * o2;
end