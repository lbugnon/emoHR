% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% LICENSING 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

clear all
clc
close all
rng(sinc(0))

addpath('somtoolbox/som/')
addpath('lib/')

% You can use the provided feature set or provide here the RECOLA dataset
% with folders 'recordings_physio' and 'ratings_gold_standard' (set
% forceFeatureExtraction=true).      
dataDir='recola/';
featDir='features/';
forceFeatureExtraction=false;

% Feature extraction if 'features' folder is empty or forced (may take some
% time). 
if length(dir(featDir))<=2 || forceFeatureExtraction
    fprintf('Feature extraction\n')
    featureExtraction([dataDir,'recordings_physio/filtered/'],[dataDir,'ratings_gold_standard/'],featDir);    
end

sessionFiles={};
dirs=dir(featDir);
for f=1:length(dirs)
    if ~dirs(f).isdir
        sessionFiles=[sessionFiles; sprintf('%s%s',featDir,dirs(f).name)];
    end
end


% RECOLA partitions
trainPartition=sessionFiles(19:27);
devPartition=sessionFiles(1:9);


results={'Classifier','Target','CCC_Train','CCC_Dev','Outputs','Reference'};
targets={'Arousal','Valence'};
classifiers={'sSOM','nELM','kELM'};

for classifier=classifiers
    classifier=classifier{:};
    for target=targets
        target=target{:};
        fprintf('Running: %s-%s\n',classifier,target);
                
        % Load hyperparameters
        load(sprintf('config/%s-%s',classifier,target(1)));
        
        % Gen train/test partitions =======================================
       
        % Train/dev
        trainData0=makeDataStruct(trainPartition,parameters);
        % Original labels were used for the competence. As our features are
        % framed at 2hz, we use a subsampled version of labels.
        testData0=makeDataStruct(devPartition,parameters,'includeOriginal',1);
        
        % pre-process features
        [trainData,testData]=preProcessing(trainData0,testData0,parameters);
        
        % train and estimate 
        switch classifier
            case 'sSOM'
                model=SOMTrain(trainData,parameters);
                [trainOutput]=SOMPredict(trainData,model,parameters);
                [testOutput]=SOMPredict(testData,model,parameters);
            case {'nELM','kELM'}
                model=ELMTrain(trainData,parameters);
                [trainOutput]=ELMPredict(trainData,model,parameters);
                [testOutput]=ELMPredict(testData,model,parameters);
        end

        % Post processing 
        [trainOutputFilt,testOutputFilt]=postProcessing(trainOutput,...
           trainData0,testOutput,parameters);
        
        % Interpolation of test output for validation with original
        % reference. 
        % (Features sampling is 2hz and original labels sampling is 25hz)
        for s=1:length(testData0.sessionFiles)
            
            firstval=testOutputFilt{s}(1,:);
            lastval=testOutputFilt{s}(end,:);
   
            time=[testData0.timeStamp(s,1):1/testData0.fs_feat:testData0.timeStamp(s,2)]';
            timeOriginal=0:1/25:300;
            
            filtOutput=interp1(time,testOutputFilt{s},timeOriginal)';
            % Replace NaN in extrapolation
            for n=1:size(filtOutput,1)
                if isnan(filtOutput(n,1)) && n<size(filtOutput,1)/2
                    filtOutput(n,:)=firstval;
                end
                if isnan(filtOutput(n,1)) && n>=size(filtOutput,1)/2
                    filtOutput(n,:)=lastval;
                end
            end
            
            testOutputFilt{s}=filtOutput;
        end
     
        
        % Stats ===========================================================
        trainRes=cccPerformance(trainOutputFilt,trainData0);
        [testRes,ref]=cccPerformance(testOutputFilt,testData0,'original',1);
        % Stats ===========================================================
        
        results=[results; {classifier,target,trainRes,testRes,testOutputFilt,ref}];
    end
end


fprintf('===============================\n')
fprintf('Classifier   Arousal    Valence\n')
for k=2:2:size(results,1)
    fprintf('%s         %0.3f      %0.3f\n',results{k,1},results{k,4},results{k+1,4})
end
fprintf('===============================\n')
fprintf('##DONE##\n')

% =========================================================================
% Plot outputs and reference
set(0,'defaultfigurecolor',[1 1 1])
time=0:1/25:300;
for t=targets
    figure('Name',t{:},'Position', get(0,'Screensize'))
    for s=1:9
        session=devPartition{s}(10:end-4);
        subplot(3,3,s)
        ref=results(strcmp(results(:,2),t),end);
        plot(time,ref{1}{s},'black','LineWidth',1)
        hold all
        for c=classifiers
            output=results(strcmp(results(:,1),c) & strcmp(results(:,2),t),end-1);
            plot(time,output{1}{s})
        end
        title(sprintf('Session: %s',session))
    end
    xlabel('Time [s]')
    ylabel('Mean Rating')
    suptitle(t{:})
    legend(['Ref',classifiers])
end
   
% =========================================================================
% Graphical sSOM: train a sSOM for arousal-valence space and plot features
% and target layers. 
figure('Position', get(0,'Screensize'))
load('config/sSOM-2D.mat')
showFeat=[1 7 21 25 2];
trainData0=makeDataStruct([trainPartition,devPartition],parameters);

trainData=preProcessing(trainData0,testData0,parameters);

model=SOMTrain(trainData,parameters);
som_show(model.som,'comp',[showFeat,29,30],'footnote','','subplots',[2 4])
suptitle('Features and target layers')