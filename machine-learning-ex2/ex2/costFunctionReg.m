function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
k = size(X,2);
logisf = -y .* log(h) - (1-y) .* log(1-h);
J = (1/m) * sum(logisf) + (1/(2*m)) * lambda * (sum (theta .^ 2) - theta(1) .^2);
% for j = 1 : k
%     if j = 1;
%         grad(j) = sum((h(:,1) - y) .* X(:,1))/m;
%     else 
%         grad(j) = sum((h(:,1) - y) .* X(:,j))/m + lambda * theta(j)/m;
%     end

grad(1) = sum((h(:,1) - y) .* X(:,1))/m;

for j = 2 : k
    grad(j) = sum((h(:,1) - y) .* X(:,j))/m + lambda * theta(j)/m;
    
    
% =============================================================

end
