load('test_data');
train_fht = X_fht(:,:,1:floor(size(X_fht,3)*0.8));
train_yt = y_t(1:floor(size(X_fht,3)*0.8));

test_fht = X_fht(:,:,ceil(size(X_fht,3)*0.8):end);
test_yt = y_t(ceil(size(X_fht,3)*0.8):end);

%% train a model based on test data
% [theta,train_err] = fit_NRF_model(X_fht,y_t,minibatch_number,...
%    regtype,lam,net_str,num_pass,theta_init)

[theta,train_err]=fit_NRF_model(train_fht,train_yt,30,'abs',1e-5,{20 1},20);

% for crossvalidation use v2
% [theta,train_err,optim_lam,MSE]=fit_NRF_model_v2(train_fht,...
%     train_yt,30,'abs');
%% now use the model to make prediction
v_hat = NRF_model(test_fht,theta);

%% plot the results
% error function
subplot(5,2,1)
loglog(train_err)
xlabel('iteration')
ylabel('err')

% data and prediction
subplot(5,2,2)
hold on
plot(test_yt,'b');
plot(v_hat,'r');
legend('data','prediction');

% hidden units sorted by variance of their weight matrix
[~,ind] = sort(var(theta{2},[],2),'descend');

for ii=1:length(ind)
  II=ind(ii);
  subplot(5,5,5+ii)
  weights = reshape(theta{2}(II,:),size(X_fht,1),size(X_fht,2));
  weights = weights * sign(theta{1}(II));
  maxabs = max(abs(weights(:)));
  imagesc(weights,[-maxabs maxabs]);
  axis xy;
end
