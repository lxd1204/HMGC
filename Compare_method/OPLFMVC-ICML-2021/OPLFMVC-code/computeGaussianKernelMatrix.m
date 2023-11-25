function K = computeGaussianKernelMatrix(X, sigma)
    % 计算高斯核矩阵
    % 参数：
    %   X: 输入数据矩阵，每一行代表一个数据样本
    %   sigma: 高斯核函数的参数

    % 获取数据样本的数量
    m = size(X, 1);

    % 初始化高斯核矩阵
    K = zeros(m, m);

    % 计算高斯核矩阵的每一个元素
    for i = 1:m
        for j = 1:m
            % 计算欧几里得距离的平方
            distSq = sum((X(i, :) - X(j, :)).^2);
            
            % 计算高斯核函数的值
            K(i, j) = exp(-distSq / (2 * sigma^2));
        end
    end
end
