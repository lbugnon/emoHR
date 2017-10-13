% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Lin's concordance correlation coeficient. Outputs is a cell of sessions
% targets. testData contains target reference. 
function [stats reftargets]=cccPerformance(output,testData,varargin)
original=false;
% aditional parameters
for v=1:2:length(varargin)
    switch varargin{v}
        case 'original'
            original=varargin{v+1};
    end
end

if isempty(testData.targets)
    stats=NaN;
    reftargets=[];
    return
end
Ns=size(testData.sessionLabels,1);
stats=zeros(1,size(output{1},2));
reftargets=cell(Ns,1);

% For each target dimension, get concordance score
for l=1:length(stats)
    
    res=[];
    for s=1:Ns
        if original
            expectedOutput=testData.original.targets(testData.original.sessionLabels{s,2}:...
                testData.original.sessionLabels{s,3},l);
        else
            expectedOutput=testData.targets(testData.sessionLabels{s,2}:...
                testData.sessionLabels{s,3},l);
        end
        res=[res linConcordance(output{s}(:,l),expectedOutput)];
    end
    stats(l)=mean(res);  
end

for s=1:Ns
    if original
        reftargets{s}=testData.original.targets(testData.original.sessionLabels{s,2}:...
            testData.original.sessionLabels{s,3},:);
    else
        reftargets{s}=testData.targets(testData.sessionLabels{s,2}:testData.sessionLabels{s,3},:);
    end
end
end
% =========================================================================
function [concord pearson]=linConcordance(x,y)

concord=zeros(size(y,2),1);
pearson=concord;

for k=1:length(concord)
    stdx=std(x);
    stdy=std(y(:,k));
    mux=mean(x);
    muy=mean(y(:,k));
    N=length(x);
    
    pearson(k)=(x-mux)'*(y-muy)/N/stdx/stdy;
    concord(k)=2*(x-mux)'*(y-muy)/N/(stdx^2+stdy^2+(mux-muy)^2);
end

end
