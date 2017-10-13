% Modified version from Support Vector Regression
%  by Ronnie Clark
% ================================================
% sinc(i) - http://sinc.unl.edu.ar
% Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
% ================================================
function f = svr_predict(y,svrmodel)
   
f = 0;   
    
for n=1:size(svrmodel.trainData,1)
    % gauss kernel 
    f = f + svrmodel.alpha(n)*gaussKernel(y,svrmodel.trainData(n,:),svrmodel.gamma);
end
    
f = f + svrmodel.b;
f = f/2;
end