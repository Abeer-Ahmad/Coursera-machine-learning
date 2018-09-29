function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predict = X * theta; % m-dimentional col vector of predicted outputs; applying the hypothesis function
err = predict - y; % m-dimensional col vector of differences between the predicted outputs and real values
theta_reg = theta(2:length(theta));
J = ((1 / (2 * m)) * err' * err) + ((lambda / (2 * m)) * (theta_reg' * theta_reg));
grad = ((1 / m) * X' * err) + ((lambda / m) * theta); % updating ALL thetas (including theta0 [theta(1)])
grad(1) = (X(:,1)' * err) / m; % excluding the gradient of theta0 from the regularized form

% =========================================================================

grad = grad(:);

end
