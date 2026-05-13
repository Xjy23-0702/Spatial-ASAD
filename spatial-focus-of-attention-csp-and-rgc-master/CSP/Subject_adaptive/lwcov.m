function Rxx = lwcov(X)
% LWCOV Compute a well-conditioned (regularized) covariance matrix
% according to the method by Ledoit & Wolf [1], i.e., by computing a new
% covariance matrix as a weighted combination of the (poor-conditioned, but
% unbiased) sample covariance matrix and the (well-conditioned, but
% uninformative) identity matrix (= shrinkage).
% - The regularized covariance is the optimal linear shrinkage in the
% minimum mean squared error sense: it minimizes the expected squared
% error deviation from the true but unknown covariance, and as such is at
% the optimal point of the 'bias-variance' tradeoff that is omnipresent in
% statistics and machine learning (cfr. eq. (10) in [1]).
% - Since a linear combination with an identity matrix is made, the
% eigenvectors (~ principal subspaces) of the covariance are not altered,
% but only their corresponding eigenvalues are shrunk towards their mean.
% This is necessary since eigenvalues estimated from a finite data sample
% are too dispersed w.r.t. the ground truth.
% - In summary, the new covariance matrix is both more accurate (!) and
% better conditioned than than the sample covariance matrix.
%
%   Input parameters:
%       X [DOUBLE]: a data matrix (observations x variables)
%
%   Output parameters:
%       Rxx [DOUBLE]: regularized covariance matrix (variables x variables)
%
% [1] Ledoit, Olivier, and Michael Wolf. "A well-conditioned estimator for
% large-dimensional covariance matrices." Journal of multivariate analysis
% 88.2 (2004): 365-411.

% Author: Simon Van Eyndhoven, KU Leuven, ESAT


[nobs,nvar] = size(X);

X = bsxfun(@minus,X,mean(X,1));%对每一列减去其均值，使数据零均值。

assert(nobs>1) % at least two observations needed
S = (1/(nobs-1))*(X'*X);%标准无偏样本协方差矩阵（除以前面是 nobs-1）。

m = trace(S)/nvar; %  所有特征值的均值
d2 = (norm(S-m*eye(nvar),'fro').^2)/nvar; % 总的方差（分散度）
b2 = min(calcbbar2(X,S),d2); % 估计的真实协方差矩阵的“偏差平方”
a2 = d2 - b2; % 收缩后的方差
Rxx = b2/d2 * m * eye(nvar) + a2/d2 * S; % 计算正则化协方差
end

function bbar2 = calcbbar2(X,S)


[nobs,nvar] = size(X);
rownorms = sum(X.^2,2);
term11 = X'*bsxfun( @times , X , rownorms ); % first squared term
term12 = -2*S*(X'*X); % cross-term
term22 = nobs*(S*S'); % second squared term

bbar2 = trace(term11 + term12 + term22)/(nvar*nobs^2);
end