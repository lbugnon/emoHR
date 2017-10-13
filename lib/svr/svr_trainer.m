% Modified version from Support Vector Regression
%  by Ronnie Clark
% ================================================
% sinc(i) - http://sinc.unl.edu.ar
% Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
% ================================================

function svrobj = svr_trainer(xdata,ydata, c, epsilon, kernel, varargin)
% SVR  Utilises Support Vector Regression to approximate
%           the functional relationship from which the
%           the training data was generated.
%  Function call:
%
%    svrobj = svr_trainer(x_train,y_train,c,epsilon,kernel,varargin);
%    The training data, x_train and y_train must be column vectors.
%


if strcmp(kernel,'gaussian')
    gamma = varargin{1};
    %kernel_function = @(x,y) exp(-lambda*norm(x-y,2)^2);
    kernel_function = @gaussKernel;
elseif strcmp(kernel,'spline')
    kernel_function = @(a,b) prod(arrayfun(@(x,y) 1 + x*y+x*y*min(x,y)-(x+y)/2*min(x,y)^2+1/3*min(x,y)^3,a.feature,b.feature));
elseif strcmp(kernel,'periodic')
    l = varargin{1};
    p = varargin{2};
    kernel_function = @(x,y) exp(-2*sin(pi*norm(x.feature-y.feature,2)/p)^2/l^2);
elseif strcmp(kernel,'tangent')
    a = varargin{1};
    c = varargin{2};
    kernel_function = @(x,y) prod(tanh(a*x.feature'*y.feature+c));
end

ntrain = size(xdata,1);

alpha0 = zeros(ntrain,1);

M=gaussKernel(xdata,xdata,gamma);

% *********************************
% Set up the Gram matrix for the
% training data.
% *********************************
%M = arrayfun(kernel_function,xi,xj);
M = M + 1/c*eye(ntrain);

% *********************************
% Train the SVR by optimising the
% dual function ie. find a_i's
% *********************************
% remove verbosity with 'Display','off'
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
H = sparse(0.5*[M zeros(ntrain,3*ntrain); zeros(3*ntrain,4*ntrain)]);

%figure; imagesc(M); title('Inner product between training data (ie. K(x_i,x_j)'); xlabel('Training point #'); ylabel('Training point #');

lb = sparse([-c*ones(ntrain,1);	zeros(ntrain,1);	zeros(2*ntrain,1)]);
ub = sparse([ c*ones(ntrain,1);	2*c*ones(ntrain,1); c*ones(2*ntrain,1)]);
f = sparse([ -ydata; epsilon*ones(ntrain,1);zeros(ntrain,1);zeros(ntrain,1)]);
H = quadprog(H,f,[],[],[],[],lb,ub,[],options);

alpha = H(1:ntrain);
%figure; stem(alpha); title('Visualization of the trained SVR'); xlabel('Training point #'); ylabel('Weight (ie. alpha_i - alpha_i^*)');
% *********************************
% Calculate b
% *********************************
for m=1:ntrain
    bmat(m)=ydata(m)-sum(alpha'.*M(m,:))- epsilon - alpha(m)/c;
end
b = mean(bmat);

% *********************************
% Store the trained SVR.
% *********************************
svrobj.alpha = alpha;
svrobj.b = b;
svrobj.kernel=kernel;
svrobj.gamma=gamma;
svrobj.trainData = xdata;

end

