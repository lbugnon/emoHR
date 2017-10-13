% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Methods for Extreme Learning Machine training (classification and
% regression).  
function model=ELMTrain(trainData,parameters,varargin)

for p=1:length(parameters)
   switch parameters(p).name
       case 'elmMethod'
           elmMethod=parameters(p).val;
   end
   
end

switch elmMethod
    case 'neural'
        model=elmNeuralTrain(trainData.features, trainData.targets, parameters);
    case 'kernel'
        model=elmKernelTrain(trainData.features, trainData.targets, parameters);        
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
% Train ELM Neural model for classification or regression. TrainLabels is
% [NxD], trainFeat is [NxM]
function [model, output] =elmNeuralTrain(feat,labels,parameters,binarized)


for p=1:length(parameters)
    switch parameters(p).name
        case 'task'
            task=parameters(p).val;
        case 'nhid'
            nhid=parameters(p).val;
        case 'neuralFunc'
            neuralFunc=parameters(p).val;
    end
end

if nargin<4
    binarized=false;
end

NumberofTrainingData=size(feat,1);
NumberofInputNeurons=size(feat,2);

classes=[];
if isequal(task,'classification') && ~binarized
    % Labels must be converted to binary expressions, so 1 is [-1,1]
    % and 2 is [1,-1] in case of 2 classes. In this case, [Nx1] is
    % converted to [Nx2]...if labels are [NxD], must convert to [Nxcomb(classes_d1,classes_d2)]
    
    [labels,classes]=binarizeLabels(labels);
end

% Random generate input weights InputWeight (w_i) and biases
% BiasofHiddenNeurons (b_i) of hidden neurons

% nhidx
InputWeight=rand(nhid,NumberofInputNeurons)*2-1;
% nhidx1
BiasofHiddenNeurons=rand(nhid,1);
% nhidxM*MxN

tempH=InputWeight*feat';

clear feat;                         %   Release input of training data
%   Extend the bias matrix BiasofHiddenNeurons to match the dimention of H
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
[H,funcList]=activationFunction(tempH,neuralFunc);
% H is nhidxN

%   Release the temparary array for calculation of hidden neuron output matrix H
clear tempH;
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
% This is the solution of minimize ||Y-w2*H||² with minimum ||w2||.
try
    OutputWeight= labels'*pinv(H);
catch ME
    % In case the pseudo inverse cant converge
    OutputWeight= zeros(size(labels,2),nhid);
    fprintf('#\n#\nWarning: pseudo-inverse not found\n#\n')
end
%%%%%%%%%%% Train output
output=(OutputWeight*H)';
clear H;

model=struct;
model.elmMethod='neural';
if nhid>0
    model.w1=InputWeight;
    model.b1=BiasofHiddenNeurons;
end
model.w2=OutputWeight;
model.neuralFunc=neuralFunc;
model.task=task;
model.funcList=funcList;
if isequal(task,'classification')
    if isempty(classes)
        classes=unique(labels,'rows');
    end
    model.classes=classes;    
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
% Train ELM model with kernel function. TrainLabels is [NxD], trainFeat is
% [NxM]. 
function model =elmKernelTrain(feat, labels,parameters)

for p=1:length(parameters)
    switch parameters(p).name
        case 'task'
            task=parameters(p).val;
        case 'kernelType'
            kernelType=parameters(p).val;
        case 'kernelParam1'
            kernelParam1=parameters(p).val;
        case 'kernelParam2'
            kernelParam2=parameters(p).val;
        case 'kernelC'
            C=parameters(p).val;
        case 'kernelParams' % this is a parametrized form of kernelParam1 and kernelC
            C=parameters(p).val(1);            
            kernelParam1=parameters(p).val(2);
            if length(parameters(p).val)==3
                kernelParam2=parameters(p).val(3);
            else
                kernelParam2=[];
            end
    end
end

NumberofTrainingData=size(feat,1);
NumberofInputNeurons=size(feat,2);

if isequal(task,'classification')
    % Labels must be converted to binary expressions, so 1 is [-1,1]
    % and 2 is [1,-1] in case of 2 classes. In this case, [Nx1] is
    % converted to [Nx2]...if labels are [NxD], must convert to [Nxcomb(classes_d1,classes_d2)]
    
    [labels,classes]=binarizeLabels(labels);
end

Omega_train = kernel_matrix(feat,kernelType, [kernelParam1 kernelParam2]);
try
    OutputWeight=((Omega_train+speye(size(feat,1))/C)\(double(labels)));
catch ME
    % In case the ¿pseudo inverse? cant converge
    OutputWeight= zeros(size(labels,2),1)';
    fprintf('#\n#\nWarning: kernel sol not found\n')
    fprintf('%s\n#\n#\n',ME.message)
end
model=struct;
model.elmMethod='kernel';
model.task=task;
model.wo=OutputWeight;
model.kernelType=kernelType;
model.kernelParam1=kernelParam1;
model.kernelParam2=kernelParam2;
model.trainFeat=feat;
if isequal(task,'classification')
    model.classes=classes;
end
end
% =========================================================================

% =========================================================================
function [binLabels,classes]=binarizeLabels(labels)
classes=unique(labels,'rows');
binLabels=-ones(size(labels,1),size(classes,1));    
for n=1:size(labels,1)
    for c=1:length(classes)
        if labels(n,:)==classes(c,:)
            binLabels(n,c)=1;
        end
    end
end

end