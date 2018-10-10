function [theta,train_err] = fit_crossvalidated_NRF_model(X_fht,y_t,...
    minibatch_number,regtype,lam,crossvalidation_fold,net_str,num_pass)
% [theta,train_err] = fit_NRF_model(X_fht,y_t,minibatch_number,...
%    regtype,lam,net_str,num_pass,theta_init)
%
% This is an implementation of NRF model (similar to Harper et. al. 2016)
% with sigmoid non-linearity
%
% regtype : 'abs' or 'sq'
% lam: anything from 0 to 1
% num_pass: number of iteration through data, 20 is fine
% theta_init: initial value of network parameters
% 
% Most of input options of the function have got default values.
% So the the model will work if only X_fht and y_t are set.
%
% Author: Monzilur Rahman
% Year: 2016
% monzilur.rahman@gmail.com
%

% set defaults
% Network structure
if ~exist('net_str','var')
    J= 20; % number of hidden units
    K = 1; % number of output units
else
    J = net_str{1};
    K = net_str{2};
end

if ~exist('regtype','var')
    regtype='abs'; % regularization type
end

if ~exist('lam','var')
    lam=1e-5;
end

if ~exist('crossvalidation_fold','var')
    crossvalidation_fold = 1;
end

if ~exist('num_pass')
    num_pass = 20;
end

if ~exist('minibatch_number','var')
    minibatch_number = 25;
end

% Randomize the data
rng('shuffle');
rand_ind = randperm(length(y_t));
X_fht = X_fht(:,:,rand_ind);
y_t = y_t(rand_ind);

% Make minibatches
for ii = 1:crossvalidation_fold
    % put away 10% of the data for validation
    valid_fht = X_fht(:,:,floor(size(X_fht,3)*0.9)+1:end);
    valid_yt = y_t(floor(length(y_t)*0.9)+1:end);
    train_fht = X_fht(:,:,1:floor(size(X_fht,3)*0.9));
    train_yt = y_t(1:floor(length(y_t)*0.9));
    
    batch_size = floor(length(train_yt)/minibatch_number);
    for minibatch = 1:minibatch_number
        startInd = (minibatch-1)*batch_size + 1;
        endInd = (minibatch-1)*batch_size + batch_size ;
        sub_refs{minibatch}{1} = train_fht(:,:,startInd:endInd);
        sub_refs{minibatch}{2} = train_yt(startInd:endInd);
    end

    I = size(X_fht,1)*size(X_fht,2);

    % Loop through all values of lamda
    for jj=1:length(lam)
        [theta,~] = fit_NRF_model(train_fht,train_yt,minibatch_number,...
            regtype,lam(jj),net_str,num_pass);
        y_hat = NRF_model(valid_fht,theta);
        MSE(ii,jj) = immse(y_t,y_hat);
    end  
end

% find the best lambda
[~,II] = min(mean(MSE,1));

[theta,train_err] = fit_NRF_model(X_fht,y_t,minibatch_number,...
            regtype,lam(II),net_str,num_pass);

end
