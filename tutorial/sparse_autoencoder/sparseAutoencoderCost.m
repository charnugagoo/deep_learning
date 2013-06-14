function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
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

% tmpCost = 0;
% weight = 0 ;% weight penalty
% sparse = 0 ;% sparsity penalties
% [n m] = size(data);% m is the number of samples, n is the sample number of features
% 
% %First run the forward algorithm on all data
% z2 = W1 * data;
% a2 = sigmoid(z2);
% z3 = W2 * a2;
% a3 = sigmoid(z3);
% 
% % calculated prediction error generated
% tmpCost = (0.5/m) * sum(sum((a3-data).^2));
% 
% % penalty term weights are calculated
% weight = (1/2) * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% 
% % calculated dilution Rules
% rho = (1/m) .* sum(a2, 2); %average value of hidden layer
% sparse = sum(sparsityParam .* log(sparsityParam ./ rho) + ...
%         (1 - sparsityParam) .* log((1 - sparsityParam) ./ ( 1 - rho)));
% 
% % total loss function expression
% cost = tmpCost + lambda * weight + beta * sparse;
% 
% % reverse algorithm derived error values ??for each node
% d3 = -(data-a3) .* sigmoid_derivative(z3);
% 
% % due to the additional sparse Rules, so
% %when calculating the partial derivatives of the need to introduce
% term = beta * (-sparsityParam ./ rho + (1 - sparsityParam) ./ ( 1 -rho ));
% 
% d2 = (W2' * d3 + repmat(term, 1, m)) .* sigmoid_derivative(z2);
% 
% % calculated W1grad
% W1grad = W1grad + d2 * data'; 
% W1grad = (1 / m) * W1grad + lambda * W1;
% 
% % calculated W2grad  
% W2grad = W2grad + d3 * a2'; 
% W2grad = (1 / m) .* W2grad + lambda * W2;
% 
% % calculated b1grad
% % Note that the partial derivative of b is a vector, so here the value of each row should add up
% b1grad = b1grad + sum(d2, 2);
% b1grad = (1 / m) * b1grad;
% 
% % calculated b2grad
% b2grad = b2grad + sum(d3, 2 );
% b2grad = ( 1 / m) * b2grad;

% mm=size(data,2);
% 
% z2=(W1*data)+repmat(b1,1,mm);
% a2=sigmoid(z2);
% z3=(W2*a2)+repmat(b2,1,mm);
% a3=sigmoid(z3);
% 
% rho=sum(a2,2)./mm;
% 
% sparse = sum(sparsityParam .* log(sparsityParam ./ rho) + ...
%          (1 - sparsityParam) .* log((1 - sparsityParam) ./ ( 1 - rho)));
% 
% cost=1/2/mm*sum(sum((a3-data).^2))+lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)))+beta*sum(KLDiv(rho,sparsityParam));
% sparityy=(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
% 
% x=data();
% delta3=-(x-a3).*sigmoid_derivative(z3);
% delta2=(W2'*delta3+repmat(beta*sparityy,1,mm)).*a2.*(1-a2);
% W1grad=W1grad+delta2*x';
% W2grad=W2grad+delta3*a2';
% b1grad=b1grad+sum(delta2,2);
% b2grad=b2grad+sum(delta3,2);
% 
% b1grad=b1grad/mm;
% b2grad=b2grad/mm;
% W1grad=W1grad/mm+lambda*W1;
% W2grad=W2grad/mm+lambda*W2;




datasize = size(data);
numpatches = datasize(2);

% Row-vector to aid in calculation of hidden activations and output values
weightsbuffer = ones(1, numpatches);

% Calculate activations of hidden and output neurons
hiddeninputs = W1 * data + b1 * weightsbuffer; % hiddensize * numpatches
hiddenvalues = sigmoid( hiddeninputs ); % hiddensize * numpatches

finalinputs = W2 * hiddenvalues + b2 * weightsbuffer; %visiblesize * numpatches
outputs = sigmoid( finalinputs ); %visiblesize * numpatches

% Least squares component of cost
errors = outputs - data; %visiblesize * numpatches
%leastsquares = power(norm(errors), 2) / (2 * numpatches); % Average least squares error over numpatches samples

leastsquares = sum( sum((errors .* errors)) ./ (2*numpatches));

% Back-propagation calculation of gradients
delta3 = errors .* outputs .* (1 - outputs); % Matrix of error terms, visiblesize * numpatches
W2grad = delta3 * transpose(hiddenvalues) / numpatches; % visiblesize * hiddensize, averaged over all patches
b2grad = delta3 * transpose(weightsbuffer) / numpatches; % visiblesize * 1, averaged over all patches

% Sparsity stuff
avgactivations = hiddenvalues * transpose(weightsbuffer) / numpatches; % hiddensize * 1
sparsityvec = -sparsityParam ./ avgactivations + (1 - sparsityParam) ./ (1 - avgactivations); % hiddensize * 1
% sparsityvec * weightsbuffer; % Add this to the delta2 parenthesis
kldiv = sparsityParam * log(prod(sparsityParam ./ avgactivations)) + (1 - sparsityParam) * log(prod( (1 - sparsityParam) ./ (1 - avgactivations) )); % Add this to cost

delta2 = (transpose(W2) * delta3 + beta * sparsityvec * weightsbuffer) .* hiddenvalues .* (1 - hiddenvalues); % hiddensize * numpatches
W1grad = delta2 * transpose(data) / numpatches; % hiddensize * visiblesize, averaged over all patches
b1grad = delta2 * transpose(weightsbuffer) / numpatches; % hiddensize * 1, averaged over all patches

cost = leastsquares + beta * kldiv;


% Weight-decay
%%cost = cost + lambda / 2 * ( power(norm(W1), 2) + power(norm(W2), 2) );

cost = cost + lambda / 2 * ( sum(sum(W1 .* W1)) +sum( sum(W2 .* W2)) );

W1grad = W1grad + lambda * W1;
W2grad = W2grad + lambda * W2;



%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end




%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

% derivative of sigmoid function
function sigmInv = sigmoid_derivative(x)
    sigmInv = sigmoid (x).* ( 1 - sigmoid (x));
end


function dist=KLDiv(P,Q)
    %  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
    %  distributions
    %  P and Q  are automatically normalised to have the sum of one on rows
    % have the length of one at each 
    % P =  n x nbins
    % Q =  1 x nbins or n x nbins(one to one)
    % dist = n x 1



    if size(P,2)~=size(Q,2)
        error('the number of columns in P and Q should be the same');
    end

    if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
       error('the inputs contain non-finite values!') 
    end

    % normalizing the P and Q
    if size(Q,1)==1
        Q = Q ./sum(Q);
        P = P ./repmat(sum(P,2),[1 size(P,2)]);
        temp =  P.*log(P./repmat(Q,[size(P,1) 1]));
        temp(isnan(temp))=0;% resolving the case when P(i)==0
        dist = sum(temp,2);


    elseif size(Q,1)==size(P,1)

        Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
        P = P ./repmat(sum(P,2),[1 size(P,2)]);
        temp =  P.*log(P./Q);
        temp(isnan(temp))=0; % resolving the case when P(i)==0
        dist = sum(temp,2);
    end
end


