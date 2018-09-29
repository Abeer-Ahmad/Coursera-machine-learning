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

h = sigmoid(X * theta); % m-dimentional col vector of predicted outputs; applying the sigmoid/logistic hypothesis function
err = h - y; % m-dimensional col vector of difference between the predicted outputs and real values
theta_reg = theta(2:length(theta));
J = (-(y' * log(h) + (1 - y)' * log(1 - h)) + (0.5 * lambda * (theta_reg' * theta_reg))) / m; % adding the regularization term (0.5 * ...)
grad = (X' * err + lambda * theta) / m; % updating ALL thetas (including theta0 [theta(1)])
grad(1) = (X(:,1)' * err) / m; % excluding the gradient of theta0 from the regularized form

% =============================================================

end
