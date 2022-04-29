k=5;
RUNS = 1;
NumOfReps = 16; % sparse303=32 ring238=32 sparse622=64

EvalHitRate = zeros(RUNS,10);
EvalAriRate = zeros(RUNS,10);
EvalEdgesPercent = zeros(RUNS,10);
EvalRuntime = zeros(RUNS,10);
ColPos = 1;

FlagPlot = false;
FlagPlotResult = false;
FlagNormalize = false;

PRE_Points;

AffMatMethod = {'kmeans','LocalSigma';
    'kmeans','CONN';
    'SOM','CONN';
    'kmeans','CONNHybrid';
    'SOM','CONNHybrid'};    
for i=1:size(AffMatMethod,1)    
    for run=1:RUNS
        tic
        RUN_Points_VQ(X, k, AffMatMethod{i,1}, NumOfReps, FlagPlot, AffMatMethod{i,2}, FlagNormalize)        
        EvalRuntime(run,ColPos) = toc;
        POST_Points;
        EvalHitRate(run,ColPos) = hitrate;
        EvalAriRate(run,ColPos) = arirate;
        EvalEdgesPercent(run,ColPos) = edgesPercent;    
        close all;
        disp([' this was run = ' num2str(run) ' out of ' AffMatMethod{i,1} ' ' AffMatMethod{i,2}]);
    end
    ColPos = ColPos + 1;
end

for i=[3 7]
    for run=1:RUNS
        tic
        RUN_Points_Fast_Old(X, k, i, FlagPlot, FlagPlotResult, FlagNormalize);
        EvalRuntime(run,ColPos) = toc;
        POST_Points;
        EvalHitRate(run,ColPos) = hitrate;
        EvalAriRate(run,ColPos) = arirate;
        EvalEdgesPercent(run,ColPos) = edgesPercent;    
        close all;
        disp([' this was run = ' num2str(run) ' out of our old method with LocalSigmaK7 = ' num2str(i)]);
    end
    ColPos = ColPos + 1;
end

for run=1:RUNS
    tic
    RUN_Points_Fast(X, k, FlagPlot, FlagPlotResult, FlagNormalize);
    EvalRuntime(run,ColPos) = toc;
    POST_Points;
    EvalHitRate(run,ColPos) = hitrate;
    EvalAriRate(run,ColPos) = arirate;
    EvalEdgesPercent(run,ColPos) = edgesPercent;    
%     close all;
    disp([' this was run = ' num2str(run) ' out of our method']);
end
