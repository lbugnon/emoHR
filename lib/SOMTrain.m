% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Methods for training supervised self-organization maps (sSOM).
function model=SOMTrain(trainData,parameters)

tracking=0;
somTopologyRatio=0.5;
lambda=ones(size(trainData.targets,2),1);
mapInit='lininit';
training='short';
for p=1:length(parameters)
    switch parameters(p).name
        case 'mapsize'
            mapsize=parameters(p).val;
        case 'somTopologyRatio'
            somTopologyRatio=parameters(p).val;
        case 'lambda'
            lambda=parameters(p).val;
    end
end

% Build train struct ======================================================
model=struct;
model.dtargets=length(trainData.targetNames); % Dimension of labels

[trainStruct,ok,msg]=som_set('som_data','data',[trainData.features ...
    bsxfun(@times,lambda,double(trainData.targets))],...
    'comp_names',[trainData.featNames,trainData.targetNames],...
    'comp_norm',cell(1,length(trainData.featNames)+model.dtargets));

if ~isempty(find(ok==0))
    errorMsg=msg(ok==0);
    error(errorMsg{1})
end
% =========================================================================


% Train SOM ===============================================================
somTopology=[mapsize floor(mapsize*somTopologyRatio)];

model.som=som_make(trainStruct,'init',mapInit,'msize',somTopology,...
    'training',training,'tracking',tracking);
% =========================================================================

end