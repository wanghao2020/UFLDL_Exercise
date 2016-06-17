function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes   分类的数量
% inputSize - the size N of the input vector  输入向量的个数
% lambda - weight decay parameter       权重的衰减项系数
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set            N * M 的数据集
% labels - an M x 1 matrix containing the labels corresponding for the input data
%                                       对应的数据的类别

% Unroll the parameters from theta  还原theta的原来的格式
theta = reshape(theta, numClasses, inputSize);
% data = inputData;
% data有多少类
numCases = size(data, 2);

% 记录最后的结果，每一列对应所识别的类别
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta * data;
%  减去最大项，防止溢出
M = bsxfun(@minus,M,max(M,[],1));
% H = H ./ repmat(sum(H),size(H,1),1)
p = exp(M) ./ repmat(sum(exp(M)),numclasses,1);











% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

