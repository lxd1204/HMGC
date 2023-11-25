% 加载Iris数据集
clear;
addpath(genpath(pwd));
load('3sources.mat');
X = full(fea{3,1});

% 使用K均值算法进行聚类（假设我们知道聚类数为3）
rng(42,'twister') % 设置随机数种子，以便结果可重复
[idx, centers] = kmeans(X, 3);

% 使用PCA进行数据降维，以便在二维空间中进行可视化
coeff = pca(X);
X_pca = X * coeff(:, 1:2);

% 创建散点图
figure;
gscatter(X_pca(:, 1), X_pca(:, 2), idx, 'rgb', 'osd');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('K-Means Clustering of Iris Dataset');

% 添加图例
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Location', 'best');

% 显示图形
grid on;