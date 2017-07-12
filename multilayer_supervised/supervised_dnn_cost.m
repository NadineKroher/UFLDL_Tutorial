function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%   Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% aux functions
sigmoid = @(x)1./(1+exp(-x));
smax = @(x)bsxfun(@rdivide,exp(x),sum(exp(x)));
tanh = @(x)(exp(x)-exp(-x))./(exp(x)+exp(-x));
relu = @(x)max(0,x);

%% forward prop
switch ei.activation_fun
    case 'logistic'
        actFun = sigmoid;
    case 'tanh'
        actFun = tanh;
    case 'relu'
        actFun = relu;
    otherwise
        disp('unknown activation function, using logistic...')
        actFun = sigmoid;
end

hAct{1} = actFun(bsxfun(@plus,data'*stack{1}.W',stack{1}.b')); %input layer
for l = 2 : length(stack)-1 % hidden layers excl. softmax
    hAct{l} = actFun(bsxfun(@plus,hAct{l-1}*stack{l}.W',stack{l}.b'));
end
% softmax layer
hAct{end} = smax(bsxfun(@plus,hAct{end-1}*stack{end}.W',stack{end}.b')')';
pred_prob = hAct{end};

%% return here if only predictions desired.
if po
  cost = -1; %ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  pred_prob = pred_prob';
  return;
end;

%% compute cost
I = sub2ind(size(pred_prob),1:size(pred_prob,1)',labels');
p = pred_prob(I);
lp = log(p);
ceCost = -sum(lp);
  
%% compute gradients using backpropagation
% softmax layer
S = zeros(size(pred_prob));
S(I) = 1;
P = S-pred_prob;
errTerm = cell(length(gradStack),1);
errTerm{end} = -P';
gradStack{end}.W = errTerm{end}*hAct{end-1};
gradStack{end}.b = mean(errTerm{end},2);
% hidden layers
for i = length(errTerm)-1 : -1 : 1
    errTerm{i} = ((stack{i+1}.W'*errTerm{i+1}).*(hAct{i}.*(1-hAct{i}))');
    if i > 1
        gradStack{i}.W = errTerm{i}*hAct{i-1};
    else
        gradStack{1}.W = errTerm{1}*data';
    end
    gradStack{i}.b = mean(errTerm{i},2);
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for l = 1:numHidden+1
    wCost = wCost + .5 * ei.lambda * sum(stack{l}.W(:).^2);
end
cost = ceCost + wCost;
pred_prob = pred_prob';

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



