% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Create a struct from data in sessionFiles.   
function dataStruct=makeDataStruct(sessionFiles,parameters,varargin)

includeOriginal=false;
allFeatures=true;


for v=1:2:length(varargin)
   switch varargin{v}
       case 'featSelected'
           featSelected=varargin{v+1};
           allFeatures=false;
       case 'includeOriginal'
           includeOriginal=varargin{v+1};
   end
     
end


for p=1:length(parameters)
    switch parameters(p).name
        case 'selectedLabels'
            targetsSelected=parameters(p).val;
        case 'classificationTask'
            classificationTask=parameters(p).val;
    end
end


dataStruct=struct;
dataStruct.features=[];
dataStruct.targets=[];
dataStruct.timeStamp=[];
dataStruct.sessionLabels={};
if includeOriginal
    dataStruct.original=struct;
    dataStruct.original.targets=[];
    dataStruct.original.sessionLabels={};
    dataStruct.original.timeStamp=[];
end


targetNames={'Arousal','Valence'};
dataStruct.targetNames=targetNames(targetsSelected);

for s=sessionFiles'
    s=s{:};
    load(s)
    
    if allFeatures
        featSelected=1:size(features,2);
    end
    
    dataStruct.features=[dataStruct.features; features(:,featSelected)];

    try
        dataStruct.targets=[dataStruct.targets; targets(:,targetsSelected)];
    catch
    end
    session={s(strfind(s,'/')+1:end-4)};
    if isempty(dataStruct.sessionLabels) 
        dataStruct.sessionLabels=[dataStruct.sessionLabels; [session 1 size(features,1)]];
    else
        dataStruct.sessionLabels=[dataStruct.sessionLabels; [session ...
            dataStruct.sessionLabels{end,3}+1 dataStruct.sessionLabels{end,3}+size(features,1)]];
    end
    
    dataStruct.timeStamp=[dataStruct.timeStamp; timeStamp(1) timeStamp(end)];
    
    dataStruct.includeOriginal=includeOriginal;
    try
        if includeOriginal
            dataStruct.original.targets=[dataStruct.original.targets; originalTargets(:,targetsSelected)];
            
            if isempty(dataStruct.original.sessionLabels)
                dataStruct.original.sessionLabels=[dataStruct.original.sessionLabels; {session 1 size(originalTargets,1)}];
            else
                dataStruct.original.sessionLabels=[dataStruct.original.sessionLabels; {session ...
                    dataStruct.original.sessionLabels{end,3}+1 dataStruct.original.sessionLabels{end,3}+size(originalTargets,1)}];
            end
            
            dataStruct.original.timeStamp=[dataStruct.original.timeStamp; originalTimeStamp(1) originalTimeStamp(end)];
            dataStruct.original.fs_feat=fs_targets;
            
        end
    catch
    end
end

dataStruct.featNames=featNames;
dataStruct.fs_feat=fs_feat;
dataStruct.sessionFiles=sessionFiles;
dataStruct.targetsSelected=targetsSelected;

end