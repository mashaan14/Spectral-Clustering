%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML111
% Project Title: Neural Gas Network in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function net = NeuralGasNetwork(X, params, PlotFlag)

    if ~exist('PlotFlag','var')
        PlotFlag = false;
    end

    %% Load Data
    
    nData = size(X,1);
    nDim = size(X,2);

    X = X(randperm(nData),:);
    S = RandStream.getGlobalStream;
    FlagQE = evalin('base', 'FlagQE');
    BestQE = inf;
    CounterQE = 0;

    %% Parameters
    N = params.N;
    MaxIt = params.MaxIt;
    tmax = params.tmax;
    epsilon_initial = params.epsilon_initial;
    epsilon_final = params.epsilon_final;
    lambda_initial = params.lambda_initial;
    lambda_final = params.lambda_final;
    T_initial = params.T_initial;
    T_final = params.T_final;

    %% Initialization
%     % PC Initialization: scatter neurons linearly over the first two principle components
%     XLimit = linspace(min(X(:,1)),max(X(:,1)),round(sqrt(N)));
%     YLimit = linspace(min(X(:,2)),max(X(:,2)),round(sqrt(N)));
%     [GridX,GridY]=meshgrid(XLimit,YLimit);
%     w=[GridX(:) GridY(:)];
%     coeff = pca(X,'NumComponents',2);
%     w = w * coeff;
    
    % k-means++ Initialization: Select the first seed randomly and the rest of the seeds by a probabilistic model
    [w(1,:), ~] = datasample(S,X,1,1);
    minDist = inf(N,1);
   
   for ii = 2:N
        [minDist,~] = pdist2(w,X,'squaredeuclidean','Smallest',1);
%         minDist = min(minDist,distfun(X,C(:,ii-1),'sqeuclidean'));
        denominator = sum(minDist);
        if denominator==0 || isinf(denominator) || isnan(denominator)
            w(:,ii:N) = datasample(S,X,N-ii+1,2,'Replace',false);
            break;
        end
        sampleProbability = minDist/denominator;
        sampleProbability = sampleProbability';
        [w(ii,:), ~] = datasample(S,X,1,1,'Replace',false,...
            'Weights',sampleProbability);        
    end

    C = zeros(N, N);
    t = zeros(N, N);

    tt = 0;

    %% Main Loop
    for it = 1:MaxIt
        % Slect Input Vector as per a probabilistic model created by k-means++
        [minDist,~] = pdist2(w,X,'squaredeuclidean','Smallest',1);
        denominator = sum(minDist);
        if denominator==0 || isinf(denominator) || isnan(denominator)
            x = datasample(S,X,1,2,'Replace',false);
        end
        sampleProbability = minDist/denominator;
        sampleProbability = sampleProbability';
        for r = 1:400
            [x, ~] = datasample(S,X,1,1,'Replace',false,'Weights',sampleProbability);

            % Competion and Ranking
            d = pdist2(x,w);
            [~, SortOrder] = sort(d);

            % Calculate Parameters
            epsilon = epsilon_initial*(epsilon_final/epsilon_initial)^(tt/tmax);
            lambda = lambda_initial*(lambda_final/lambda_initial)^(tt/tmax);
            T = T_initial*(T_final/T_initial)^(tt/tmax);

            % Adaptation
            for ki = 0:N-1
                i=SortOrder(ki+1);
                w(i,:) = w(i,:) + epsilon*exp(-ki/lambda)*(x-w(i,:));
            end
            tt = tt + 1;

            % Creating Links
            i = SortOrder(1);
            j = SortOrder(2);
            C(i,j) = 1;
            C(j,i) = 1;
            t(i,j) = 0;
            t(j,i) = 0;

            % Aging
            t(i,:) = t(i,:) + 1;
            t(:,i) = t(:,i) + 1;

            % Remove Old Links
            OldLinks = t(i,:)>T;
            C(i, OldLinks) = 0;
            C(OldLinks, i) = 0;
        end

        % Plot
        if PlotFlag
            figure(1);
            PlotResults(X, w, C);
            pause(0.01);
        end
        
        % stopping criterion based on the change in mean quantization error
        if mod(it,10) == 0
            CurrentQE = mean(minDist);
            if CurrentQE < BestQE
                BestQE = CurrentQE;
            else
                CounterQE = CounterQE+1;
            end
            
            if CounterQE==10
                break;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        if FlagQE
            try
                wr = evalin('base','WeightRecord');
            catch
                wr = {};
            end
            wr{end+1,1} = w;
            assignin('base','WeightRecord',wr);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

    %% Export Results
    
    net.w = w;
    net.C = C;
    net.t = t;
    
end