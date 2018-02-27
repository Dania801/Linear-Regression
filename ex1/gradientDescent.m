function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  J_deriv0 = 0;
  for i=1:m
   h = theta(1,1) + theta(2,1)*X(i,2);
   J_deriv0 += (h - y(i, 1));
  endfor
  
  J_deriv1 = 0;
  for i=1:m
   h = theta(1,1) + theta(2,1)*X(i,2);
   J_deriv1 += (h - y(i, 1))*X(i,2);
  endfor
  
  theta1_temp = theta(1,1) - (alpha * (1/m) * J_deriv0);
  theta2_temp = theta(2,1) - (alpha * (1/m) * J_deriv1);

  theta(1,1) = theta1_temp;
  theta(2,1) = theta2_temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
