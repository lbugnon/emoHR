% ================================================
% sinc(i) - http://sinc.unl.edu.ar
% Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
% ================================================
function M=gaussKernel(x,y,gamma)

XXh1 = sum(x.^2,2)*ones(1,size(y,1));
XXh2 = sum(y.^2,2)*ones(1,size(x,1));
M = XXh1+XXh2' - 2*x*y';
M = exp(-M*gamma);
end

