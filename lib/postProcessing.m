% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Generic function for postprocessing classifier output, like filtering and
% interpolation.  
function [trainOutputProcessed,testOutputProcessed]=postProcessing...
    (trainOutput,trainData,testOutput,parameters)

outputScaling='no';
for p=1:length(parameters)
    switch parameters(p).name
        case 'outputScaling'
            outputScaling=parameters(p).val;
    end
end


% Aditional processing on classifier output (e.g. filtering):
[trainOutputProcessed]=smoothingFunc(trainOutput,parameters);
[testOutputProcessed]=smoothingFunc(testOutput,parameters);

preScaling=testOutputProcessed;%debug

switch outputScaling
    case 'no'
    case 'trainAmpl'
        % get mean amplitude diference of outputs and labels (only in training)
        scaleFactor=zeros(1,size(trainOutputProcessed{1},2));
        for s=1:length(trainOutputProcessed) % session
            sind=trainData.sessionLabels{s,2}:trainData.sessionLabels{s,3};
            outputAmpl=max(trainOutputProcessed{s})-min(trainOutputProcessed{s});
            
            labelAmpl=max(trainData.targets(sind,:))-min(trainData.targets(sind,:));
            
            scaleFactor=scaleFactor+labelAmpl./outputAmpl;
        end
        % finish average
        scaleFactor=scaleFactor/length(trainOutputProcessed);
        
        for s=1:length(trainOutputProcessed) % session
            for l=1:size(trainOutputProcessed{s},2)
                trainOutputProcessed{s}(:,l)=(trainOutputProcessed{s}(:,l)-mean(trainOutputProcessed{s}(:,l)))*scaleFactor(l)+mean(trainOutputProcessed{s}(:,l));
            end
        end
        for s=1:length(testOutputProcessed) % session
            for l=1:size(testOutputProcessed{s},2)
                testOutputProcessed{s}(:,l)=(testOutputProcessed{s}(:,l)-mean(testOutputProcessed{s}(:,l)))*scaleFactor(l)+mean(testOutputProcessed{s}(:,l));
            end
        end
end
    
end
% =========================================================================

% =========================================================================
function [output]=smoothingFunc(input,parameters,varargin)

delay=0;
filterFunc='no';
for p=1:length(parameters)
    switch parameters(p).name
        case 'filterFunc'
            % function and parameter combined
            filterFunc=parameters(p).val{1};
            filterParam=parameters(p).val{2};
        case 'delay'
            delay=parameters(p).val;
    end
end

output=input;

% Process each session 
for s=1:length(output)
    
    switch filterFunc
        case 'median'
            output{s}=medfilt1(output{s},filterParam);
        case 'average'
            outputcopy=output{s};
            for n=filterParam/2:size(output{s},1)-filterParam/2
                output{s}(n,:)=mean(outputcopy((n-filterParam/2+1):(n+filterParam/2),:));
            end
        case 'exp'
            output{s}=smoothts(output{s}','e',filterParam)';
        case 'no'
            % do nothing
    end
    
    % Label delay =========================================================
    if delay~=0
        output{s}=circshift(output{s},delay);
    end
    % =============================================================
     
end
end
            