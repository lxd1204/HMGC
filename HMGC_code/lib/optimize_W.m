function [W] = optimize_W(A)
%OPTIMIZE_W 此处显示有关此函数的摘要
%   此处显示详细说明
maxiter=20;
[n,~]=size(A);
V=(A+A')./2;
mu
for i=1:maxiter
    B=1./(1+mu./2).*(A-Lambda+mu./2.*V);
    for j=1:n
     %   x=EuclideanPro(ones(1,n),B(j,:));
        x=ProjectOntoSimplex(B(j,:),1);
        W(j,:)=x;
    end
end

end

