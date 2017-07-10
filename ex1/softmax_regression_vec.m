function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  theta(:,num_classes) = 0;
  
  tx = theta'*X;
  P = bsxfun(@rdivide,exp(tx),sum(exp(tx)));
  lP = log(P);
  I = sub2ind(size(lP),y,1:size(lP,2));
  p = lP(I);
  p = p(y<10);
  f = -sum(p);
  

  S = zeros(size(P));
  S(I) = 1;
  P = S-P;
  g = -X*P';
  g(:,10) = [];
  g=g(:); 

