function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  
  % initialize objective value and gradient.
  h = @(z)(1./(1+exp(-z)));
  f = -sum((y.*log(h(theta'*X))+(1-y).*log(1-h(theta'*X))));
  y_hat = theta'*X; % so y_hat(i) = theta' * X(:,i).  Note that y_hat is a *row-vector*.
  g = X*(y_hat - y)';

