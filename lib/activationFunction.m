%%%%    Authors:    MR HONG-MING ZHOU AND DR GUANG-BIN HUANG
%%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
%%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
%%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%%%%    DATE:       MARCH 2012
function [H,funcOrderOut]=activationFunction(X,func,funcOrderIn)

if nargin==2
    funcOrderIn=[];
end
funcOrderOut=[];

switch lower(func)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-X));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(X);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(X);            
    case {'lin'}
        %%%%%%%% Linear
        H = X;
    case {'tanh'}
        H = tanh(X);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        
    case {'rand'}
        H=X;
        for x=1:length(X)
            if isempty(funcOrderIn) % generate functions
                func=randi(5,1);
                funcOrderOut=[funcOrderOut; func];
            else % use model functions
                func=funcOrderIn(x);
            end
            switch func
                case 1
                    %%%%%%%% Sigmoid
                    H(x) = 1 ./ (1 + exp(-X(x)));
                case 2
                    %%%%%%%% Sine
                    H(x) = sin(X(x));
                case 3
                    %%%%%%%% Hard Limit
                    H(x) = hardlim(X(x));
                    %%%%%%%% More activation functions can be added here
                case 4
                    %%%%%%%% Hard Limit
                    H(x) = X(x);
                case 5
                    H(x) = tanh(X(x));
            end
        end
end
end

function a=hardlim(n)
a = double(n >= 0);
a(isnan(n)) = nan;
end
