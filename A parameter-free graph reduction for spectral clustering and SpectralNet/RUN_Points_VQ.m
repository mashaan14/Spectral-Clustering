function RUN_Points_VQ(X, k, method, NumOfNeurons, FlagPlot, AffMatMethod, FlagNormalize)

    FlagNormalize = true;
    assignin('base', 'FlagPlot', FlagPlot);
%     XFull = X;
%     [X,ia,ic] = unique(X,'rows');
    [NumOfRow,NumOfCol] = size(X);
    MaxK = 20;    
    %% =======================================================
    if FlagNormalize
        % normalize dataset to zero mean and unit variance
        NormalizeCoeff1 = zeros(2,NumOfCol);
        for r=1:NumOfCol
            NormalizeCoeff1(1,r) = mean(X(:,r)); NormalizeCoeff1(2,r) = std(X(:,r));
            % zero-mean by removing the average and unit variance by dividing by the standard deviation
            X(:,r) = (X(:,r)-mean(X(:,r))) / std(X(:,r));
        end
    end
    SOMFirstDimension1 = round(sqrt(NumOfNeurons)); SOMFirstDimension2 = round(sqrt(NumOfNeurons));
    % =======================================================
    switch method
        case 'SOM'
            % Create a Self-Organizing Map
            dimension1 = SOMFirstDimension1;
            dimension2 = SOMFirstDimension2;
            net = selforgmap([dimension1 dimension2]);
%             net.trainFcn = 'Mytrainbu';
            net.trainParam.epochs = 200;
            NeuronsDistances = net.layers{1}.distances; NeuronsDistances(NeuronsDistances~=1) = 0;
            assignin('base', 'NeuronsDistances', NeuronsDistances);
            % Train the Network
            [net,tr] = train(net,X');
            
            if FlagPlot
                figure, plotsomnd(net);
                figure, plotsomhits(net,X');
            end
            NeuronsWeights = net.IW{1,1};
            [~,LabelsInitial] = pdist2(NeuronsWeights,X,'euclidean','Smallest',1);
            LabelsInitial = LabelsInitial';

        case 'NG'
            % Create and Train Neural Gas Network
            params.N = NumOfNeurons;
            params.MaxIt = 100;
            params.tmax = 10000;
            params.epsilon_initial = 0.5;%0.4;
            params.epsilon_final = 0.05;%0.02;
            params.lambda_initial = 10;%2;
            params.lambda_final = 0.01;%0.1;
            params.T_initial = 5;
            params.T_final = 10;
            net = NeuralGasNetwork(X, params, FlagPlot);
            
            NeuronsWeights = net.w;
            NeuronsDistances = net.C;
            [~,LabelsInitial] = pdist2(NeuronsWeights,X,'euclidean','Smallest',1);
            LabelsInitial = LabelsInitial';            

        case 'GNG'
            % Create and Train Growing Neural Gas Network
            params.N = NumOfNeurons;
            params.MaxIt = 100;
            params.L = 50;
            params.epsilon_b = 0.2;
            params.epsilon_n = 0.006;
            params.alpha = 0.5;
            params.delta = 0.995;
            params.T = 50;
            net = GrowingNeuralGasNetwork(X, params, FlagPlot);
            
            NeuronsWeights = net.w;
            NeuronsDistances = net.C;
            [~,LabelsInitial] = pdist2(NeuronsWeights,X,'euclidean','Smallest',1);
            LabelsInitial = LabelsInitial';
            
        case 'kmeans'
            [idx,C] = kmeans(X,NumOfNeurons,'maxiter',200,'replicates',5,'EmptyAction','singleton');
            LabelsInitial = idx;
            NeuronsWeights = C;
            [NumOfRowNeurons,NumOfColNeurons] = size(NeuronsWeights);
            NeuronsDistances = ones(NumOfNeurons,NumOfNeurons);
    end    
    [NumOfRowNeurons,NumOfColNeurons] = size(NeuronsWeights);
    clearvars -except X ia ic k LabelsInitial AffMatMethod SigmaLocal NumOfNeurons NumOfRow NumOfCol FlagPlot FlagPlotResult FlagNormalize MaxK NumOfRowNeurons NumOfColNeurons NeuronsWeights NeuronsDistances NormalizeCoeff1
    %% =======================================================
    if FlagNormalize
        % denormalize dataset to original space
        for r=1:NumOfColNeurons
            % zero-mean by removing the average and unit variance by dividing by the standard deviation
            NeuronsWeights(:,r) = (NeuronsWeights(:,r)+NormalizeCoeff1(1,r)) * NormalizeCoeff1(2,r);
            X(:,r) = (X(:,r)+NormalizeCoeff1(1,r)) * NormalizeCoeff1(2,r);
        end
    end
    %% =======================================================
    % build the affinity matrix
    SigmaLocal = 5;
    switch AffMatMethod
        case 'LocalSigma'            
            % as per Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering.", 2005.
            % the fowlloing distance should be the distance between two samples not squared
            [D,~] = pdist2(NeuronsWeights,NeuronsWeights,'euclidean','Smallest',SigmaLocal);
            D=D';
            SigmaLocalMatrix = D(:,SigmaLocal);
            NeuronsDistancesSparse = sparse(NeuronsDistances);
            [i,j,~] = find(NeuronsDistancesSparse);
            v = zeros(size(i));
            for r=1:length(i)
                % as per Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering.", 2005.
                % the fowlloing distance should be the square distance between two samples
                NeuronsWeightsDifference = -1 * pdist2(NeuronsWeights(i(r),:),NeuronsWeights(j(r),:),'squaredeuclidean');
                v(r) = NeuronsWeightsDifference / (SigmaLocalMatrix(i(r),1) * SigmaLocalMatrix(j(r),1));
                v(r) = exp(v(r));
            end
            LocalSigmaGraph = graph(i,j,v);
            LocalSigmaGraph = simplify(LocalSigmaGraph,'mean');
            AffGraph = LocalSigmaGraph;
        case 'CONN'
            [~,NeuronsShared] = pdist2(NeuronsWeights,X,'euclidean','Smallest',2);
            NeuronsShared = NeuronsShared';
            NeuronsShared = [NeuronsShared ones(NumOfRow,1)];
            NeuronsSharedMatrix = sparse(NeuronsShared(:,1),NeuronsShared(:,2),NeuronsShared(:,3),NumOfRowNeurons,NumOfRowNeurons);
            [i,j,~] = find(NeuronsSharedMatrix);
            v = zeros(size(i));
            for r=1:length(i)
                v(r) = NeuronsSharedMatrix(i(r),j(r)) + NeuronsSharedMatrix(j(r),i(r));
            end
            CONNGraph = graph(i,j,v);
            CONNGraph = simplify(CONNGraph,'sum');
            AffGraph = CONNGraph;
        case 'CONNHybrid'
            [D,~] = pdist2(NeuronsWeights,NeuronsWeights,'euclidean','Smallest',SigmaLocal);
            D=D';
            SigmaLocalMatrix = D(:,SigmaLocal);
            [~,NeuronsShared] = pdist2(NeuronsWeights,X,'euclidean','Smallest',2);
            NeuronsShared = NeuronsShared';
            NeuronsShared = [NeuronsShared ones(NumOfRow,1)];
            NeuronsSharedMatrix = sparse(NeuronsShared(:,1),NeuronsShared(:,2),NeuronsShared(:,3),NumOfRowNeurons,NumOfRowNeurons);
            [i,j,~] = find(NeuronsSharedMatrix);
            v = zeros(size(i));
            for r=1:length(i)
                v(r) = NeuronsSharedMatrix(i(r),j(r)) + NeuronsSharedMatrix(j(r),i(r));
            end
            CONNGraph = graph(i,j,v);
            CONNGraph = simplify(CONNGraph,'sum');            
            CONNEdgeList = table2array(CONNGraph.Edges);
            CONNMax = max(CONNEdgeList(:,3));
            v = zeros(size(CONNEdgeList,1),1);
            for r=1:size(CONNEdgeList,1)
                i = CONNEdgeList(r,1);
                j = CONNEdgeList(r,2);
                NeuronsWeightsDifference = -1 * pdist2(NeuronsWeights(i,:),NeuronsWeights(j,:),'squaredeuclidean');
                v(r) = exp(NeuronsWeightsDifference / (SigmaLocalMatrix(i,1) * SigmaLocalMatrix(j,1)));
                v(r) = v(r) * exp(CONNEdgeList(r,3)/CONNMax);
            end
            CONNHybridGraph = graph(CONNEdgeList(:,1),CONNEdgeList(:,2),v);            
            AffGraph = CONNHybridGraph;
    end    
    EdgesPercent = (numedges(AffGraph) / (NumOfRow*(NumOfRow-1)/2)) * 100;
    assignin('base', 'edgesPercent', EdgesPercent);
    assignin('base', 'edgesNum', numedges(AffGraph));
    
    % draw the corresponding graph
    GraphEdgeWidth = 5;
    LWidths = GraphEdgeWidth*AffGraph.Edges.Weight/max(AffGraph.Edges.Weight);
    if FlagPlot
        figure; hold on;
        plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
        plot(AffGraph,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
            'EdgeLabel',[],'LineWidth',LWidths,...
            'NodeLabel',[],'MarkerSize',5,'NodeColor',[0.9500 0.9000 0.2500],...
            'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2],'EdgeFontSize',18);
        hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
    end
    clearvars -except X ia ic k LabelsInitial NumOfNeurons NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK NumOfRowNeurons NumOfColNeurons AffGraph
    %% =======================================================
    % The steps for spectral clustering are taken from:
    % Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems. 2002.    
    % 1 - Form the affinity matrix A
    % 2 - Define D the diagonal matrix
    AffGraphEdgesHalf = table2array(AffGraph.Edges);
    AffGraphEdges = [AffGraphEdgesHalf; [AffGraphEdgesHalf(:,2) AffGraphEdgesHalf(:,1) AffGraphEdgesHalf(:,3)]];
    AffMat = sparse(AffGraphEdges(:,1),AffGraphEdges(:,2),AffGraphEdges(:,3),NumOfRowNeurons,NumOfRowNeurons);
    N = size(AffMat,1);
    % D
    AffMatDeg = full(sum(AffMat,2));
    % D^(-1/2)
    AffMatDegN = 1./sqrt(AffMatDeg+eps);
    % sparse D^(-1/2)
    DN = sparse(1:N,1:N,AffMatDegN);
    clear AffMatDeg AffMatDegN
    eyeN = sparse(speye(N));
    
    % 3 - Find k largest eigenvectors of Laplacian L to form a matrix X
    LapN = eyeN - DN * AffMat * DN;
    clear eyeN DN AffMat
    [~,D,vN] = svds(LapN,min(NumOfNeurons,30),'smallest');
    % a work around if the graph Laplacian is not symmetric and the eigen
    % solver produces complex numbers, trying to symmetrize the Laplacian is not producing good results.
    if ~isreal(D)
        disp('The eigen solver produces complex numbers because the graph Laplacian is not symmetric, only the real part will be considered');
        D = real(D);
        vN = real(vN);
    end
    lambda=diag(D);
    [ls, is] = sort(lambda,'ascend');
    vNSort = vN(:,is);
    
    if k==0
        if sum(ls==0) > 1
            k0 = find(ls==0, 1, 'last');
        else
            k0 = 2;
        end
        disp(['k0 = ' num2str(k0)]);
        k = k0;
        for r=k0+1:length(ls)
            lsMeanNew = mean(ls(k0:r+1));
            lsMeanOld = mean(ls(k0:r));
            lsStd = std(ls(k0:r));
            if (lsMeanOld+lsStd) < lsMeanNew
                break;
            end        
            k = k+1;
            if r > MaxK
                if any(ls==0)
                    disp(['STD went to far, I am setting k to number of zeros']);
                    k = k0;
                else
                    disp(['STD went to far, I am setting k to the largest difference']);
                    [~,k] = max(abs(diff(ls(1:MaxK))));
                end
                break;
            end
        end
    end
    disp(['k = ' num2str(k)]);

    % a plot to illustrate the eigengap see:
    % - Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.
    % - https://math.stackexchange.com/questions/1248131/unequal-numbers-of-eigenvalues-and-eigenvectors-in-svd
    if FlagPlot
        figure; hold on;
        plot(ls(1:end),'o','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1]); %ylim([0 ls(end)]);
        plot(k,ls(k),'o','color',[0.6350, 0.0780, 0.1840],'MarkerFaceColor',[0.6350, 0.0780, 0.1840],'MarkerSize',10);
        xlabel('eigenvectors','FontSize',18); ylabel('\lambda','FontSize',18); % set(gca,'xtick',[],'ytick',[]);
    end               
    kerN = vNSort;
    
    % 4 - Form the matrix Y by normalizing X
    normN = sum(kerN .^2, 2) .^.5;
    kerNS = bsxfun(@rdivide, kerN, normN + eps);
    clearvars -except X ia ic k LabelsInitial NumOfNeurons NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK NumOfRowNeurons NumOfColNeurons NeuronsWeights kerNS AffGraphEdgesHalf   
    %% =======================================================
    LabelsDecimal = zeros(size(kerNS,1),k);
    for r=2:k
        LabelsDecimal(:,r) = kmeans(kerNS(:,2:r),k,'maxiter',500,'replicates',3,'EmptyAction','singleton');
    end
    
    % IntraClusterWeights is the sum of all edges where vertices are in the same class
    % InterClusterWeights is the sum of all edges where vertices are in the different classes
    % CoherenceIndex for a certain eigenvector = IntraClusterWeights / InterClusterWeights
    CoherenceIndex = zeros(1,k);
    for r=2:k
        LocalSigmaGraph2Edges1 = AffGraphEdgesHalf;
        LocalSigmaGraph2Edges1(:,1) = LabelsDecimal(LocalSigmaGraph2Edges1(:,1),r);
        LocalSigmaGraph2Edges1(:,2) = LabelsDecimal(LocalSigmaGraph2Edges1(:,2),r);
        if r==2; CoherenceIndexAccWeightsAll = sum(LocalSigmaGraph2Edges1(:,3)); end
        if ~isempty(LocalSigmaGraph2Edges1(LocalSigmaGraph2Edges1(:,1)~=LocalSigmaGraph2Edges1(:,2)))
            CoherenceIndex(r) = sum(LocalSigmaGraph2Edges1(LocalSigmaGraph2Edges1(:,1)~=LocalSigmaGraph2Edges1(:,2),3));
        else
            CoherenceIndex(r)  = min(LocalSigmaGraph2Edges1(:,3));
        end
    end
    CoherenceIndex = (CoherenceIndex / CoherenceIndexAccWeightsAll) * 100;
    CoherenceIndex(1)=inf;
    LabelsBestIndex = find(CoherenceIndex==min(CoherenceIndex), 1, 'first' );
    disp(['LabelsBestIndex = ' num2str(LabelsBestIndex)]);
    
    if FlagPlot
        figure; hold on;
        plot(3:length(CoherenceIndex),CoherenceIndex(3:end),'o','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1]);
        plot(LabelsBestIndex,CoherenceIndex(LabelsBestIndex),'o','color',[0.6350, 0.0780, 0.1840],'MarkerFaceColor',[0.6350, 0.0780, 0.1840],'MarkerSize',10);
        xlabel('number of clusters','FontSize',18); ylabel('Coherence Index','FontSize',18);
    end
        
    LabelsBest = LabelsDecimal(:,LabelsBestIndex);            
    LabelsFinal01 = LabelsBest(LabelsInitial);
%     LabelsFinal = LabelsFinal01(ic);
    LabelsFinal = LabelsFinal01;
    if FlagPlot
        ClusterColorMap = parula;
        ClusterColorMap = ClusterColorMap(round(linspace(1,64,length(unique(LabelsFinal)))),:);
        figure; hold on;
        for r=1:size(ClusterColorMap,1)
            plot(X(LabelsFinal01==r,1),X(LabelsFinal01==r,2),'o','Color',ClusterColorMap(r,:),'MarkerFaceColor',ClusterColorMap(r,:),'MarkerSize',5);
        end
        hold off; axis off;
        pbaspect([1 1 1]); daspect([1 1 1]);
    end
    assignin('base', 'LabelsFinal', LabelsFinal);
end