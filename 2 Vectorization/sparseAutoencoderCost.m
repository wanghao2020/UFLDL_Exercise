function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64)     可视化节点数
% hiddenSize: the number of hidden units (probably 25)   隐含层节点数
% lambda: weight decay parameter                                    衰减权重
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p")  稀疏系数.
% beta: weight of sparsity penalty term                                系数权重
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

%　转化数据格式

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
a = 1

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

[n , m ] = size(data);

% 初始化过程
z2 = size(hiddenSize,m);
z3 = size(visibleSize ,m);
a1 = size(n,m);
a2 = size(hiddenSize,m);
a3 = size(visibleSize,m);

y = data;
a1 = data;

%计算前向传播过程
z2 = W1 * a1 + repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2,1,m);
a3 = sigmoid(z3);
% 
% deltaW1 = zeros(size(W1));
% deltab1 = zeros(size(b1));
% deltaW2 = zeros(size(W2));
% deltab2 = zeros(size(b2));

%计算节点平均活跃度
rho = sparsityParam;
rho_hat = (1./m)*sum(a2,2);

sparsity_delta = - rho./ rho_hat + ( 1 - rho)./(1 - rho_hat);


%计算反向传导过程
% for i = 1 : m
%     % 计算残差
%     errorterm3 = -( y(:,i) - a3(:,i)).*sigmoidGrad(z3(:,i));
%     errorterm2 = (  W2' * errorterm3  +    beta*(-sparsityParam./pavge + (1-sparsityParam)./(1-pavge))  ).*sigmoidGrad(z2(:,i));
%     
%     % 计算每次的偏差值
%     JW1grad = errorterm2*a1(:,i)';
%     Jb1grad = errorterm2;
%     JW2grad  = errorterm3*a2(:,i)';
%     Jb2grad = errorterm3;
%      
%     % 累加每次的偏差值
%     deltaW1 = deltaW1 + JW1grad;
%     deltab1 = deltab1 + Jb1grad;
%     deltaW2 = deltaW2 + JW2grad;
%     deltab2 = deltab2 +Jb2grad;
%    
% end

%  反向传播矢量化操作

errorterm3 = - (y - a3).*sigmoidGrad(z3);
errorterm2 =( W2'*errorterm3 + beta*repmat(sparsity_delta,1,m) ).*sigmoidGrad(z2);

deltaW2 = errorterm3*a2';
deltab2 = errorterm3;
deltaW1 = errorterm2*a1';
deltab1 = errorterm2;


% 梯度更新
% W1grad = (1./m) * deltaW1 + lambda * W1;
% b1grad = (1./m)*deltab1;
% W2grad = (1./m)*deltaW2 + lambda*W2;
% b2grad = (1./m)*deltab2;

W2grad = (1./m)*deltaW2 + lambda*W2;
b2grad = (1./m)*sum(deltab2,2);
W1grad = (1./m)*deltaW1 + lambda*W1;
b1grad = (1./m)*sum(deltab1,2);

%计算损失函数
cost = 1./m *sum( (1./2)* sum( (y - a3) .^2 ) ) + (lambda / 2).*(  sum( sum(W1.^2) ) + sum( sum(W2.^2) ) ) + beta * sum( sparsityParam*log(sparsityParam./rho_hat) + (1-sparsityParam)*log( (1-sparsityParam)./(1-rho_hat) ) ); 
% 
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
b=2
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

% 支持矢量化的sigmoid函数
function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function grad = sigmoidGrad(x)

   grad =  exp(-x) ./ ( ( 1 + exp(-x)).^2);
   
end



