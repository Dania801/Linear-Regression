function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_temp = zeros(size(theta));

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
  for t=1: size(theta, 1)
    J_deriv = 0 ; 
    for i=1:m
      h = 0;
      for j=1:size(X,2)
        h += X(i,j)*theta(j,1);
      endfor
      J_deriv += (h-y(i,1))*X(i,t);
    endfor
    theta_temp(t,1) = theta(t,1) - (alpha/m)*J_deriv;
  endfor
  
  theta = theta_temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
