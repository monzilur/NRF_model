function [theta,train_err,optim_lam,MSE] = fit_NRF_model_v2(X_fht,y_t,...
    minibatch_number,regtype,lam,crossvalidation_fold,net_str,num_pass)
% [theta,train_err,optim_lam,MSE] = fit_NRF_model_v2(X_fht,y_t,...
% minibatch_number,regtype,lam,crossvalidation_fold,net_str,num_pass)
%
% This is an implementation of NRF model (similar to Harper et. al. 2016)
% with sigmoid non-linearity
% This function works both in single and multiple lamda cases
%
% regtype : 'abs' or 'sq'
% lam: anything from 0 to 1
% crossvalidation_fold: number of folds for crossvalidation
% net_str: network structure e.g {20 1}
% num_pass: number of iteration through data, 20 is fine
% 
% Most of input options of the function have got default values.
% So the the model will work if only X_fht and y_t are set.
%
% Author: Monzilur Rahman
% Year: 2016
% monzilur.rahman@gmail.com
%
tic
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
    lam=logspace(log10(1e-8),log10(1e-3),10);
end

if ~exist('crossvalidation_fold','var')
    crossvalidation_fold = 8;
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

valid_fraction = length(y_t)/crossvalidation_fold;
if valid_fraction > 0.1
    valid_fraction = 0.1;
end

I = size(X_fht,1)*size(X_fht,2);

% Crossvalidation
if length(lam)>1
    for ii = 1:crossvalidation_fold
        % put away 10% of the data for validation
        valid_ind = (ii-1)*floor(length(y_t)*valid_fraction)+1:...
            (ii-1)*floor(length(y_t)*0.1)+floor(length(y_t)*valid_fraction);
        train_ind = 1:length(y_t);
        train_ind(valid_ind) = 0;
        train_ind = train_ind(train_ind~=0);
        
        
        % Make minibatches
        batch_size = floor(length(y_t(train_ind))/minibatch_number);
        for minibatch = 1:minibatch_number
            startInd = (minibatch-1)*batch_size + 1;
            endInd = (minibatch-1)*batch_size + batch_size ;
            sub_refs{minibatch}{1} = X_fht(:,:,train_ind(startInd:endInd));
            sub_refs{minibatch}{2} = y_t(train_ind(startInd:endInd));
        end
        
        disp('Initializing training of the model ...')
        disp(['Crossvalidation fold: ',num2str(ii),' of ',...
            num2str(crossvalidation_fold)])
        
        % Loop through all values of lamda
        parfor jj=1:length(lam)
            % Initialization of network parameters
            rng('shuffle');
            C = 0.5;
            W_jk = C*2*(rand(K,J)-0.5)/sqrt(J+K);
            W_ij = C*2*(rand(J,I)-0.5)/sqrt(I+J);
            b_k = C*2*(rand(K,1)-0.5)/sqrt(J+K);
            b_j = C*2*(rand(J,1)-0.5)/sqrt(I+J);

            theta_init = {W_jk,W_ij,b_k,b_j};

            % Training
            args = {J,K,lam(jj),regtype};
            % initialize the optimizer
            optimizer = sfo(@loss_function_NRF,theta_init,sub_refs,args);
            optimizer.display=0;
            % run the optimizer for half a pass through the data
            theta = optimizer.optimize(0.5);
            % run the optimizer for another 20 passes through the data, continuing from 
            % the theta value where the prior call to optimize() ended
            theta = optimizer.optimize(num_pass);
            % check the mean squared error for prediction and validation
            % data
            y_hat = NRF_model(X_fht(:,:,valid_ind),theta);
            MSE(ii,jj) = mean((y_t(valid_ind)-y_hat).^2);
        end
        toc
    end
    % find the best lambda
    [~,II] = sort(mean(MSE,1));
    II = II(1);
else
    II = 1;
    MSE = nan;
end

optim_lam = lam(II);

% Final training step with the best lambda
% Make minibatches
disp('Final training step ...')
batch_size = floor(length(y_t)/minibatch_number);
for minibatch = 1:minibatch_number
    startInd = (minibatch-1)*batch_size + 1;
    endInd = (minibatch-1)*batch_size + batch_size ;
    sub_refs{minibatch}{1} = X_fht(:,:,startInd:endInd);
    sub_refs{minibatch}{2} = y_t(startInd:endInd);
end
        
% Initialization of network parameters
rng('shuffle');
C = 0.5;
W_jk = C*2*(rand(K,J)-0.5)/sqrt(J+K);
W_ij = C*2*(rand(J,I)-0.5)/sqrt(I+J);
b_k = C*2*(rand(K,1)-0.5)/sqrt(J+K);
b_j = C*2*(rand(J,1)-0.5)/sqrt(I+J);

theta_init = {W_jk,W_ij,b_k,b_j};

% Training
args = {J,K,optim_lam,regtype};
% initialize the optimizer
optimizer = sfo(@loss_function_NRF,theta_init,sub_refs,args);
% run the optimizer for half a pass through the data
theta = optimizer.optimize(0.5);
% run the optimizer for another 20 passes through the data, continuing from 
% the theta value where the prior call to optimize() ended
theta = optimizer.optimize(num_pass);

train_err = optimizer.hist_f_flat;

end
