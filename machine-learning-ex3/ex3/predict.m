function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% ============================================================

% Add ones to the X data matrix
A1 = [ones(m, 1) X]; % m*(S1=n+1) input matrix, with each row seen as a1 (activation of layer 1) for a specific example
% ============================================================
Z2 = A1 * Theta1'; % m*S2 matrix of the hidden layer; where:
                   % S2: number of units in layer 2
                   % Theta1': (S1+1=n+1)*S2
                  
A2 = sigmoid(Z2); % m*S2 activation matrix of the hidden layer, with each row seen as a2 (activation of layer 2) for a specific example
A2 = [ones(m, 1) A2]; % m*(S2+1); adding the bias unit
% ============================================================
Z3 = A2 * Theta2'; % m*S3 matrix of the output layer; where:
                   % S3: number of units in layer 3 = num_lables
                   % Theta2': (S2+1)*S3
                   
A3 = sigmoid(Z3); % m*(S3=num_lables) output matrix of the output layer, with each row seen as output vector for a specific example
% ============================================================
[max_prob indices] = max(A3, [], 2);
p = indices;

% =========================================================================


end
