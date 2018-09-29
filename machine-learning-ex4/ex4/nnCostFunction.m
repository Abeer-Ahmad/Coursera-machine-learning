function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
A1 = [ones(m, 1) X]; % m*(S1=n+1) input matrix, with each row seen as a1 (activation of layer 1) for a specific example
% ==================
Z2 = A1 * Theta1'; % m*S2 matrix of the hidden layer; where:
                   % S2: number of units in layer 2
                   % Theta1': (S1+1=n+1)*S2
                  
A2 = sigmoid(Z2); % m*S2 activation matrix of the hidden layer, with each row seen as a2 (activation of layer 2) for a specific example
A2 = [ones(m, 1) A2]; % m*(S2+1); adding the bias unit
% ===================
Z3 = A2 * Theta2'; % m*S3 matrix of the output layer; where:
                   % S3: number of units in layer 3 = num_lables
                   % Theta2': (S2+1)*S3
                   
A3 = sigmoid(Z3); % m*(S3=num_lables) output matrix of the output layer, with each row seen as output vector for a specific example
% ============================================================
Y = zeros(m, num_labels); % creating an output matrix; where each row shpuld be a binary vector with
                          % 1 at the only correct label index, and 0's otherwise
i = sub2ind(size(Y), 1:m, y'); % obtaining the linear indices
Y(i) = 1;

J = (-1 / m) * sum(sum(Y .* log(A3) + (1 - Y) .* log(1 - A3), 2)); % providing the summation dimension is not necessary (only multiplication matters);
                                                                   % summing along any dimension will yield the same result
% ============================================================
Theta1_reg = Theta1(:, 2:end); % hidden_layer_size*input_layer_size
Theta2_reg = Theta2(:, 2:end); % num_labels*hidden_layer_size

reg_term = (lambda / (2 * m)) * (sum(sum(Theta1_reg .^ 2)) + sum(sum(Theta2_reg .^ 2)));
J += reg_term;
% ============================================================
Delta1 = zeros(size(Theta1)); % hidden_layer_size*(input_layer_size+1)
Delta2 = zeros(size(Theta2)); % num_labels*(hidden_layer_size+1)

for t = 1:m
  a1 = A1(t,:)'; % (input_layer_size+1)*1 -- col vector
  a2 = A2(t,:)'; % (hidden_layer_size+1)*1 -- col vector
  a3 = A3(t,:)'; % num_labels*1 -- col vector
  
  y = Y(t,:)'; % num_labels*1 -- col vector
  delta3 = a3 - y; % num_labels*1 -- col vector
  
  z2 = [1; Z2(t,:)']; % (hidden_layer_size+1)*1 -- col vector
  delta2 = Theta2' * delta3 .* sigmoidGradient(z2); % (hidden_layer_size+1)*1 -- col vector
  delta2 = delta2(2:end); % hidden_layer_size*1 -- col vector
  
  Delta1 += delta2 * a1';
  Delta2 += delta3 * a2';
end

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;
% ============================================================
Theta1_grad += (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1_reg]; % adding a zero column so that activation units are not regularized
Theta2_grad += (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2_reg]; % adding a zero column so that activation units are not regularized

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
