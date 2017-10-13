% =========================================================================
% sinc(i) - http://fich.unl.edu.ar/sinc/
% Copyright 2017 Leandro Bugnon 
% lbugnon@sinc.unl.edu.ar
% =========================================================================
% Feature extraction from HR signal.  
function featureExtraction(hrDir,targetDir,outDir)

sessions={};
dirs=dir([hrDir]);
for f=1:length(dirs)
    if ~dirs(f).isdir
        sessions=[sessions; dirs(f).name(1:end-4)]; % remove extension
    end
end

% Constant definitions =======================
ws=20;
overlap=ws-0.5;

data=importdata(sprintf('%s%s.csv',hrDir,sessions{1}));
time_phys=data.data(:,1);
fs_phys=(time_phys(2)-time_phys(1))^-1;

window=ws*fs_phys;
nfft=window;
noverlap=floor(overlap*fs_phys);
% fs/2=N*df/2 => df=fs/N
lfIndex=ceil(window/fs_phys*0.04)+1;
hfIndex=ceil(window/fs_phys*0.15)+1;
hhfIndex=ceil(window/fs_phys*0.5)+1;
fcIndex=ceil(window/fs_phys*40)+1;      % Max freq into account
% Bands up to 1 hz
spectBands=round(linspace(1,ceil(window/fs_phys*1)+1,5));

featNames={'meanHR','ampHR','skew','kurt','LF','HF','LFHF','totalP',...
    'Pregr3','P0'};
for f=1:length(spectBands)-1
    featNames=[featNames sprintf('P%d',f)];
end
targetNames={'arousal','valence'};
% =============================================

for s=sessions'
    s=s{:};
    fprintf('Subject %s\n',s)
    
    % Load data and targets ================================================
    filename=sprintf('%s%s.csv',hrDir,s);
    data=importdata(filename);
    time_phys=data.data(:,1);
    fs_phys=(time_phys(2)-time_phys(1))^-1;
 
    hr=data.data(:,3);
 
    originalTargets=[];
    originalTimeStamp=[]; 
    
    % Load labels (if available)
    for l=1:length(targetNames)
        
        filename=sprintf('%s%s/%s.arff',targetDir,targetNames{l},s);
        try
            data=arff_read(filename);
            data=squeeze(struct2cell(data));
            originalTargets=[originalTargets cell2mat(data(3,:))'];
            if isempty(originalTimeStamp)
                originalTimeStamp=cell2mat(data(2,:))';
                fs_targets=(originalTimeStamp(2)-originalTimeStamp(1))^-1;
            end
        catch
            originalTargets=[];
            originalTimeStamp=[];
            fs_targets=[];
        end
    end
    % =====================================================================
    
    % Reflect edges for each session, ws/2 at begining and end of session.
    startBorder=hr(1:ws*fs_phys/2);
    endBorder=hr(end-ws*fs_phys/2+1:end);
    hr=[startBorder(end:-1:1); hr; endBorder(end:-1:1)];
    
    % targets for training are reflected as well
    targetr=[];
    try
        for l=1:length(targetNames)
            startBorder=originalTargets(1:ws*fs_targets/2,l);
            endBorder=originalTargets(end-ws*fs_targets/2+1:end,l);
            targetr=[targetr [startBorder(end:-1:1); originalTargets(:,l); endBorder(end:-1:1)]];
        end
    catch
    end
    % spectrogram 
    [~,F,~,P]=spectrogram(hr,window,noverlap,nfft,fs_phys,'yaxis');
    % Log scaling
    P=log(P);
    
    % Windowing...
    wcenter=window/2; % This is the 0 of original signal
    w=1;
   
    targets=[];
    features=[];
    timeStamp=[];
    sessionLabels=[];
    while wcenter<=length(hr)-window/2
        ind=(wcenter-floor(window/2))+1:(wcenter+floor(window/2));
        
        x=hr(ind).*hamming(length(ind));
        
        % Features in time scale
        meanHR=mean(x);
        meanHRNoWin=mean(hr(ind));
        ampHR=abs(min(x)-max(x));
        
        % High order statistics (maybe more data needed)
        skew=skewness(x);
        kurt=kurtosis(x);
        
        LF=sum(P(2:lfIndex,w));
        HF=sum(P(lfIndex:hfIndex,w));
        
        LFHF=LF/HF;
        totalP=sum(P(:,w));
        P0=P(1,w);
        for h=1:length(spectBands)-1
            eval(sprintf('P%d=sum(P(spectBands(h):spectBands(h+1),w)/P0);',h));
        end
        
        % cuadratic regression of spectral power in log scale.
        x=log(F(1:fcIndex)+1);
        y=P(1:fcIndex,w);
        p=polyfit(x,y,2);
        Pregr1=p(1);
        Pregr2=p(2);
        Pregr3=p(3);
        
        wfeat=zeros(1,length(featNames));
        for f=1:length(wfeat)
            eval(sprintf('wfeat(f)=%s;',featNames{f}));
        end
        
        
        features=[features; wfeat];
        
        % Here we take account of signal reflections, so d=wcenter/fs_phys is
        % the origin, and end-d is the end of signal
        timeStamp=[timeStamp wcenter/fs_phys-ws/2];
        if length(timeStamp)==2
            fs_feat=(timeStamp(2)-timeStamp(1))^-1;
        end
        sessionLabels=[sessionLabels s];
        
        % targets are taken from the center of the window
        try
            targetInd=ceil(ind(1)/fs_phys*fs_targets):ind(end)/fs_phys*fs_targets;
            targets=[targets; targetr(targetInd(end/2),:)];
        catch
        end
        
        wcenter=wcenter+(window-noverlap);
        w=w+1;
    end

    
    
    % .mat
    if exist('features/')~=7
        mkdir('features/')
    end
    save([outDir,s,'.mat'],'timeStamp','sessionLabels','targetNames',...
        'targets','features','featNames','originalTargets','fs_targets',...
        'originalTimeStamp','fs_feat');

  

end

end
