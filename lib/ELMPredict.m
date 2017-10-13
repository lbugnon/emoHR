% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Methods for Extreme Learning Machine prediction (classification and
% regression).  
function [outputs]=ELMPredict(testData,elmmodel,parameters)
outputs={};
% Test per session 
for s=1:size(testData.sessionLabels,1)
    
    sind=testData.sessionLabels{s,2}:testData.sessionLabels{s,3};
    
    switch elmmodel.elmMethod
        case 'neural'
            [soutput]=elmNeuralPredict(testData.features(sind,:),elmmodel);
        case 'kernel'
            [soutput]=elmKernelPredict(testData.features(sind,:),elmmodel);
    end
    
    outputs=[outputs; soutput];
    
end
end
% =========================================================================

% Modified from:
%%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
%%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
%%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
%%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%%%%    DATE:       APRIL 2004
% Get output of ELM neural model for regression/classification. 
function [output] =elmNeuralPredict(testFeat,model)

N=size(testFeat,1);

tempH=model.w1*testFeat';
clear testFeat;
%   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
ind=ones(1,N);
BiasMatrix=model.b1(:,ind);
tempH=tempH+BiasMatrix;

H=activationFunction(tempH,model.neuralFunc,model.funcList);
%   Release the temparary array for calculation of hidden neuron output
%   matrix H 
clear tempH;        

output=(model.w2*H)';                             
clear H;

if isequal(model.task,'classification')
    % de-binarize the outputs (select winning neuron in each output frame)
    output=debinarizeLabels(output,model.classes);
end
end
% =========================================================================

% =========================================================================
% Modified from: 
%%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
%%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
%%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
%%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%%%%    DATE:       APRIL 2004
% Test ELM kernel model for regression/classification.
function [output] =elmKernelPredict(testFeat,model)

N=size(testFeat,1);

Omega_test = kernel_matrix(model.trainFeat,model.kernelType, [model.kernelParam1 model.kernelParam2],testFeat);
output=Omega_test' * model.wo;

if isequal(model.task,'classification')
    % de-binarize the outputs (select winning neuron in each output frame)
    output=debinarizeLabels(output,model.classes);
end

end
% =========================================================================
function labels=debinarizeLabels(binLabels,classes)
labels=zeros(size(binLabels,1),size(classes,2));
for n=1:size(binLabels,1)
    [~,maxind]=max(binLabels(n,:));
    labels(n,:)=classes(maxind,:);
end
end