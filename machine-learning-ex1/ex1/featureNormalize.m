function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X; % m*n matrix
mu = zeros(1, size(X, 2)); % n-dimentional row vector
sigma = zeros(1, size(X, 2)); % n-dimentional row vector

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.  

mu = mean(X);
sigma = std(X);
sigma(sigma == 0) = 1; % to avoid NaN's and Inf's (some features of the training examples may have constant values,
                       % which means that the sigma vector has zeroes for those features)
X_norm = zscore(X); % equivalent to {X_norm = (X - mu) ./ sigma}

% ============================================================

end
