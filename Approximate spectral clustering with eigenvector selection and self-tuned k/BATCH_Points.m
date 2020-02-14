PRE_Points;
k=0;
RUNS = 100;
NumOfReps = 128;

hitCostEigenGap = zeros(RUNS,1);
BytesAllocateCostEigenGap = zeros(RUNS,1);
AffMatNNZCostEigenGap = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'GNG', NumOfReps, false, false, false,'SigmaLocal','CostEigenGap',k);
    POST_Points;
    hitCostEigenGap(run) = hitrate;
    BytesAllocateCostEigenGap(run) = BytesAllocate;
    AffMatNNZCostEigenGap(run) = AffMatNNZ;

    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['CostEigenGap this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end

hitCostZelnik = zeros(RUNS,1);
BytesAllocateCostZelnik = zeros(RUNS,1);
AffMatNNZCostZelnik = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'GNG', NumOfReps, false, false, false,'SigmaLocal','CostZelnik',k);
    POST_Points;
    hitCostZelnik(run) = hitrate;
    BytesAllocateCostZelnik(run) = BytesAllocate;
    AffMatNNZCostZelnik(run) = AffMatNNZ;

    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['CostZelnik this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end

hitCostDBIOverLambda = zeros(RUNS,1);
BytesAllocateCostDBIOverLambda = zeros(RUNS,1);
AffMatNNZCostDBIOverLambda = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'GNG', NumOfReps, false, false, false,'SigmaLocal','CostDBIOverLambda',k);
    POST_Points;
    hitCostDBIOverLambda(run) = hitrate;
    BytesAllocateCostDBIOverLambda(run) = BytesAllocate;
    AffMatNNZCostDBIOverLambda(run) = AffMatNNZ;

    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['CostDBIOverLambda this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end

hitCostDBIOverLambdaPCA = zeros(RUNS,1);
BytesAllocateCostDBIOverLambdaPCA = zeros(RUNS,1);
AffMatNNZCostDBIOverLambdaPCA = zeros(RUNS,1);
for run=1:RUNS
    PRE_Points;
    RUN_Points(X, 'GNG', NumOfReps, false, false, false,'SigmaLocal','CostDBIOverLambdaPCA',k);
    POST_Points;
    hitCostDBIOverLambdaPCA(run) = hitrate;
    BytesAllocateCostDBIOverLambdaPCA(run) = BytesAllocate;
    AffMatNNZCostDBIOverLambdaPCA(run) = AffMatNNZ;

    
    close all;
    clear WeightRecord WeightRecord1;
    disp(['CostDBIOverLambdaPCA this was run = ' num2str(run) ' out of ' num2str(RUNS)]);
end