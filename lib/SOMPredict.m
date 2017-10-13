% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Methods for prediction using supervised self-organization maps (sSOM).
function [outputs]=SOMPredict(testData,model,parameters,varargin)

outputs={};

for p=1:length(parameters)
    switch parameters(p).name
        case 'maxNBMUs'
            maxNBMUs=parameters(p).val;
        case 'lambda'
            lambda=parameters(p).val;
    end
end

% Generate struct for testing. Target information is ignored from testData.
[testStruct,ok,msg]=som_set('som_data','data',testData.features,...
    'comp_names',testData.featNames,'comp_norm',cell(1,length(testData.featNames)));
if ~isempty(find(ok==0))
    errorMsg=msg(ok==0)
    error(0)
end

dtargets=model.dtargets;
targetCentroids=model.som.codebook(:,end-dtargets+1:end);
% Discard label information for classification
model.som.codebook=model.som.codebook(:,1:end-dtargets);
model.som.comp_norm=model.som.comp_norm(1:end-dtargets);
model.som.comp_names=model.som.comp_names(1:end-dtargets);
model.som.mask=model.som.mask(1:end-dtargets);
    
% Find BMUS ===========================================
% Choose best N BMUs (using test features and trained SOM)
bmus=som_bmus(model.som,testStruct,1:maxNBMUs);
% =========================================================
                    
% Predict for each session ========================================
sessions=unique(testData.sessionLabels(:,1))';

for s=1:length(sessions)
    
    % Session index
    sind=testData.sessionLabels{s,2}:testData.sessionLabels{s,3};
   
    % Define SOM output based on N BMUs. It has the same dimension of
    % targets, and each element contains the l-target for each test frame.
    
    % Obtain outputs based on weighted bmus. The weights are obtained by
    % minimizing distance between the weighted BMUs and the feature
    % point.
    soutput=zeros(length(sind),size(targetCentroids,2));
     
    for n=1:length(sind) % for  each frame...
        
        % Find optimal number N of BMUs to minimize distance in the feature
        % space. 
        bestNbmu=getBestNBMUs(testData.features(sind(n),:),...
            model.som.codebook(bmus(sind(n),:),:));
        
        soutput(n,:)=getAverageBMUsOutput(testData.features(sind(n),:),...
            model.som.codebook(bmus(sind(n),:),:),...
            targetCentroids(bmus(sind(n),:),:),bestNbmu); % Continuous value
        
        
    end
    
    % Desafecting the lambda factor.
    soutput=bsxfun(@rdivide,soutput,lambda);
    
    outputs=[outputs; soutput];
    
end
end
% =========================================================================

% =========================================================================
% Find SOM output for input vector given the set of N best BMUs. Thexs labels
% average is weighted by their respective distances in the feature space.
% features is [1xF], featureCentroids [maxNBMUsxF] and labelCentroids
% [maxNBMUSxL], the N best labels asociated with this sample 
function output=getAverageBMUsOutput(features,featureCentroids,labelCentroids,nbmus) 


invNormDistance=zeros(nbmus,1);
for j=1:nbmus
    invNormDistance(j)=norm(features-featureCentroids(j,:))^-1;
end
invNormDistance=invNormDistance/sum(invNormDistance);

wMean=zeros(1,size(labelCentroids,2));
for j=1:nbmus
    lambda=1;
    if nbmus>1
        lambda=invNormDistance(j);
    end
    
    wMean=wMean+lambda*labelCentroids(j,:);
end

output=wMean;
end
% =========================================================================

% =========================================================================
% Find optimal number N of BMUs from SOM based on distance in feature 
% space. This is, select N so the weighted average of N-BestMatchingUnits A
% for a given value V minimize norm(A,V), using only the feature space (the
% visible informatino in train and test data). 
% maxN define the searching limit, if maxN=1, the algorithm is the classic 
% find the BMU.
% This function is for 1 sample. features is [1xF], centroids [maxNBMUsxF]
% are all BMU centroids.
function bestNbmu=getBestNBMUs(features,centroids) 

% Get mean BMUs in features subspace
meanBMUs=centroids;

% Distance between feat. point and 1:m-BMUs average
avgDistance=zeros(size(centroids,1),1);
for m=size(centroids,1):-1:1
    
    meanBMUs(m,:)=getAverageBMUsOutput(features,centroids,centroids,m);

    avgDistance(m)=norm( features-mean(meanBMUs(m:-1:1,:)));
end
% Select best nbmu for weighting
[~,bestNbmu]=min(avgDistance);
end
% =========================================================================
