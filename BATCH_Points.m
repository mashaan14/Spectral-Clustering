k=0;
RUNS = 1;
NumOfReps = 32; % sparse303=32 ring=32 Aggregation=64 sparse622=32
                % iris=32 wine=32 ImageSeg=40 statlog=100 penDigits=500 mGamma=1000

PlotShow = false;
HitRate = zeros(RUNS,7);
AriRate = zeros(RUNS,7);
EdgesPercent = zeros(RUNS,7);
ColPos = 1;

PRE_Points;
% PRE_Dataset;
% PRE_Image;

AffMatMethod = {'LocalSigma','CONN','CONNHybrid'};
method = {'kmeans','SOM'};
for i=1:length(AffMatMethod)
    for j=1:length(method)
        for run=1:RUNS            
            RUN_Points_VQ(X, k, method{j}, NumOfReps, PlotShow, AffMatMethod{i})
            POST_Points;
%             POST_Image;                           %------> only if the input is an image
            HitRate(run,ColPos) = hitrate;
            AriRate(run,ColPos) = arirate;
            EdgesPercent(run,ColPos) = edgesPercent;    
%             close all;
            disp([' this was run = ' num2str(run) ' out of ' AffMatMethod{i} ' ' method{j}]);
        end
        ColPos = ColPos + 1;
    end
end

for run=1:RUNS
    RUN_Points_Fast(X, k, PlotShow , PlotShow, false);
    POST_Points;
%     POST_Image;                           %------> only if the input is an image
    HitRate(run,ColPos) = hitrate;
    AriRate(run,ColPos) = arirate;
    EdgesPercent(run,ColPos) = edgesPercent;    
%     close all;
    disp([' this was run = ' num2str(run) ' out of our method']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=5;

HitRate1 = zeros(RUNS,7);
AriRate1 = zeros(RUNS,7);
EdgesPercent1 = zeros(RUNS,7);
ColPos = 1;
 
AffMatMethod = {'LocalSigma','CONN','CONNHybrid'};
method = {'kmeans','SOM'};
for i=1:length(AffMatMethod)
    for j=1:length(method)
        for run=1:RUNS            
            RUN_Points_VQ(X, k, method{j}, NumOfReps, PlotShow, AffMatMethod{i})
            POST_Points;
%             POST_Image;                           %------> only if the input is an image
            HitRate1(run,ColPos) = hitrate;
            AriRate1(run,ColPos) = arirate;
            EdgesPercent1(run,ColPos) = edgesPercent;    
%             close all;
            disp([' this was run = ' num2str(run) ' out of ' AffMatMethod{i} ' ' method{j}]);
        end
        ColPos = ColPos + 1;
    end
end

for run=1:RUNS
    RUN_Points_Fast(X, k, PlotShow , PlotShow, false);
    POST_Points;
%     POST_Image;                           %------> only if the input is an image
    HitRate1(run,ColPos) = hitrate;
    AriRate1(run,ColPos) = arirate;
    EdgesPercent1(run,ColPos) = edgesPercent;    
%     close all;
    disp([' this was run = ' num2str(run) ' out of our method']);
end