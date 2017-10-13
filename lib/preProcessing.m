% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Basic features pre-process: normalization, differential features and
% frame stacking.
function [trainData,testData]=preProcessing(trainData,testData,parameters)

featDiff=0;
kframeStacking=1;
for p=1:length(parameters)
   switch parameters(p).name
       case 'featDiff'
           featDiff=parameters(p).val;
       case 'frameStacking'
           kframeStacking=parameters(p).val;
       case 'featNorm'
           featNorm=parameters(p).val;
   end
end

% Differential features ===================================================
[trainData.features,trainData.featNames]=getDiffFeatures(trainData.features,...
    trainData.featNames,trainData.sessionLabels,featDiff);
[testData.features,testData.featNames]=getDiffFeatures(testData.features,...
    testData.featNames,testData.sessionLabels,featDiff);
% =========================================================================

% Frame stacking ==========================================================
[trainData.features,trainData.featNames]=stackFrames(trainData.features,...
    trainData.featNames,trainData.sessionLabels,kframeStacking);
[testData.features,testData.featNames]=stackFrames(testData.features,...
    testData.featNames,testData.sessionLabels,kframeStacking);
% =========================================================================

% Normalize features ======================================================
[trainData.features,testData.features]=normFeatures(trainData.features,...
    testData.features,featNorm,trainData.sessionLabels,testData.sessionLabels);
% =========================================================================
end
% =========================================================================
% Append differential features. ndiff can take 0:2 
function [diffFeatures, diffFeatNames]=getDiffFeatures(features,...
    featNames,sessionLabels,ndiff)

diffFeatNames=featNames;

if ndiff==0
    diffFeatures=features;
    return
end

diffFeatures=[];

for s=1:size(sessionLabels,1)
    
    sfeatures=features(sessionLabels{s,2}:sessionLabels{s,3},:);

    for n=1:size(sfeatures,1)
        
        if ndiff>=1
            if n-1>0
                diff1=diff([sfeatures(n-1,:);sfeatures(n,:)]);
            else
                diff1=zeros(size(sfeatures(n,:)));
            end
            if ndiff==1
                diffFeatures=[diffFeatures; sfeatures(n,:) diff1];
            end
        end
        
        if ndiff>=2
            if n-2>0
                diff2=diff([sfeatures(n-2,:);sfeatures(n-1,:);sfeatures(n,:)],2);
            else
                diff2=zeros(size(sfeatures(n,:)));
            end
            if ndiff==2
                diffFeatures=[diffFeatures; sfeatures(n,:) diff1 diff2];
            end
        end
    
    end
end

for d=1:ndiff
    diffFeatNames=[diffFeatNames strcat(featNames,sprintf('d%d',d))];
end
end
% =========================================================================

% =========================================================================
% Normalize training features with different methods. Use the normalization
% parameters from training dataset to normalize test. 
function [trainnorm,testnorm,meanVal,amplVal]=normFeatures(train,test,method,...
    trainSessionLabels,testSessionLabels)
% sessionLabels in the form {session indStart indEnd; session ...}

if nargin<4
    trainSessionLabels={1 1 length(train)};
    testSessionLabels={1 1 length(test)};
end

if size(trainSessionLabels,2)==1
    sessionsLab=trainSessionLabels;
    sessions=unique(sessionsLab)';
    trainSessionLabels=cell(length(sessions),3);
    for s=1:length(sessions)
        trainSessionLabels(s,:)={sessions(s) find(sessionsLab==sessions(s),1),... 
            find(sessionsLab==sessions(s),1,'last')};
    end
end

testnorm=[];

switch method
    case {'raw','no'}
        trainnorm=train;
        testnorm=test;
        meanVal=0;
        amplVal=1;
    case 'zscore'
        [trainnorm,meanVal,amplVal]=zscore(train);
        
        if ~isempty(test)
            testnorm=bsxfun(@minus,test,meanVal);
            testnorm=bsxfun(@rdivide,testnorm,amplVal);
        end
    case 'minmax'
        meanVal=mean(train);
        amplVal=max(train)-min(train);
        
        trainnorm=bsxfun(@minus,train,meanVal);
        trainnorm=bsxfun(@rdivide,trainnorm,amplVal);
        
        if ~isempty(test)
            testnorm=bsxfun(@minus,test,meanVal);
            testnorm=bsxfun(@rdivide,testnorm,amplVal);
        end
    case 'session_zscore'
        trainnorm=train;
        testnorm=test;
        meanVal=zeros(size(trainSessionLabels,1),size(train,2));
        amplVal=meanVal;
        for s=1:size(trainSessionLabels,1)
            sindex=trainSessionLabels{s,2}:trainSessionLabels{s,3};
            [trainnorm(sindex,:),meanVal(s,:),amplVal(s,:)]=zscore(train(sindex,:));
        end
        meanVal=mean(meanVal);
        amplVal=mean(amplVal);
        
        if ~isempty(test)
            for s=1:size(testSessionLabels,1)
                sindex=testSessionLabels{s,2}:testSessionLabels{s,3};
                testnorm(sindex,:)=zscore(test(sindex,:));
            end
        end     
    case 'session_minmax'
        trainnorm=train;
        
        meanVal=zeros(size(trainSessionLabels,1),size(train,2));
        amplVal=meanVal;
        
        for s=1:size(trainSessionLabels,1)
            sindex=trainSessionLabels{s,2}:trainSessionLabels{s,3};
            meanVal(s,:)=mean(train(sindex,:));
            amplVal(s,:)=max(trainnorm(sindex,:))-min(trainnorm(sindex,:));
            trainnorm(sindex,:)=bsxfun(@minus,train(sindex,:),meanVal(s,:));
            trainnorm(sindex,:)=bsxfun(@rdivide,trainnorm(sindex,:),...
                amplVal(s,:));
        end
       
        meanVal=mean(meanVal);
        amplVal=mean(amplVal);
        
        if ~isempty(test)
            testnorm=test;
            for s=1:size(testSessionLabels,1)
                sindex=testSessionLabels{s,2}:testSessionLabels{s,3};
                testnorm(sindex,:)=bsxfun(@minus,test(sindex,:),...
                    mean(test(sindex,:)));
                testnorm(sindex,:)=bsxfun(@rdivide,testnorm(sindex,:),...
                    max(testnorm(sindex,:))-min(testnorm(sindex,:)));
            end
        end 
end

% Check for NaN cases (e.g. when STD=0)
if sum(isnan(trainnorm))>0
    fprintf('warning: nan in trainFeat, replaced with 0\n')
end
if sum(isnan(testnorm))>0
    fprintf('warning: nan in testFeat, replaced with 0\n')
end

trainnorm(isnan(trainnorm))=0;
testnorm(isnan(testnorm))=0;

trainnorm(isinf(trainnorm))=0;
testnorm(isinf(testnorm))=0;
end
% =========================================================================

% =========================================================================
% Frame stacking: Given a list of frames of dimension F [NxF], returns a
% concatenation of K consecutive frames [Nx(F*K)]. In case of border
% frames, the same frame is replicated. K is odd. sessionLabels is
% necesary to stack only the proper session data.
function [sframes,sfeatNames]=stackFrames(frames,featNames,sessionLabels,k,step)
if k==1
    sframes=frames;
    sfeatNames=featNames;
    return
end
if nargin<5
    step=1;
end
    
sfeatNames=cell(1,length(featNames)*k);
kv=-step*(k-1)/2:step:step*(k-1)/2;
for dk=1:k
    for f=1:length(featNames)
        sfeatNames{(dk-1)*length(featNames)+f}=sprintf('%s_fr%d',...
            featNames{f},kv(dk));
    end
end

sframes=zeros(size(frames,1),size(frames,2)*k);
    
for s=1:size(sessionLabels,1)
    for i=sessionLabels{s,2}:sessionLabels{s,3}
        stack=[];
        for di=kv
            if i+di<sessionLabels{s,2} || i+di>sessionLabels{s,3}
                stack=[stack frames(i,:)];
            else
                stack=[stack frames(i+di,:)];
            end
        end
        sframes(i,:)=stack';
    end
    
end
end
% =========================================================================
