function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes   ���������
% inputSize - the size N of the input vector  ���������ĸ���
% lambda - weight decay parameter       Ȩ�ص�˥����ϵ��
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set            N * M �����ݼ�
% labels - an M x 1 matrix containing the labels corresponding for the input data
%                                       ��Ӧ�����ݵ����

% Unroll the parameters from theta  ��ԭtheta��ԭ���ĸ�ʽ
theta = reshape(theta, numClasses, inputSize);
% data = inputData;
% data�ж�����
numCases = size(data, 2);

% ��¼���Ľ����ÿһ�ж�Ӧ��ʶ������
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta * data;
%  ��ȥ������ֹ���
M = bsxfun(@minus,M,max(M,[],1));
% H = H ./ repmat(sum(H),size(H,1),1)
p = exp(M) ./ repmat(sum(exp(M)),numclasses,1);











% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

