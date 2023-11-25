function K = computeGaussianKernelMatrix(X, sigma)
    % �����˹�˾���
    % ������
    %   X: �������ݾ���ÿһ�д���һ����������
    %   sigma: ��˹�˺����Ĳ���

    % ��ȡ��������������
    m = size(X, 1);

    % ��ʼ����˹�˾���
    K = zeros(m, m);

    % �����˹�˾����ÿһ��Ԫ��
    for i = 1:m
        for j = 1:m
            % ����ŷ����þ����ƽ��
            distSq = sum((X(i, :) - X(j, :)).^2);
            
            % �����˹�˺�����ֵ
            K(i, j) = exp(-distSq / (2 * sigma^2));
        end
    end
end
