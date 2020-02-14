PRE_Points;
k=4;
RUNS = 100;
NumOfReps = 32;

hitCONN = zeros(RUNS,1);
NumOfEdgesCONN = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'kmeans+CHL', NumOfReps, false, false, false,'CONN','CostDBIOverLambdaPCA',k);
    POST_Points;
    hitCONN(run) = hitrate;
    NumOfEdgesCONN(run) = NumOfEdges;
    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['CONN this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end

hitCONNFilt = zeros(RUNS,1);
NumOfEdgesCONNFilt = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'kmeans+CHL', NumOfReps, false, false, false,'CONNFilt','CostDBIOverLambdaPCA',k);
    POST_Points;
    hitCONNFilt(run) = hitrate;
    NumOfEdgesCONNFilt(run) = NumOfEdges;
    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['CONNfilt this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end

hitEucInv = zeros(RUNS,1);
NumOfEdgesEucInv = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'kmeans+CHL', NumOfReps, false, false, false,'EucInv','CostDBIOverLambdaPCA',k);
    POST_Points;
    hitEucInv(run) = hitrate;
    NumOfEdgesEucInv(run) = NumOfEdges;
    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['EucInv this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end

hitEucScale = zeros(RUNS,1);
NumOfEdgesEucScale = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'kmeans+CHL', NumOfReps, false, false, false,'EucScale','CostDBIOverLambdaPCA',k);
    POST_Points;
    hitEucScale(run) = hitrate;
    NumOfEdgesEucScale(run) = NumOfEdges;
    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['EucScale this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end

hitAll = [hitCONN hitCONNFilt hitEucInv hitEucScale];
NumOfEdgesAll = [NumOfEdgesCONN NumOfEdgesCONNFilt NumOfEdgesEucInv NumOfEdgesEucScale];