function RUN_Points(X, method, NumOfNeurons, FlagPlot, FlagQE, FlagConsole, AffMatMethod, CostMethod, k)

    assignin('base', 'FlagPlot', FlagPlot);
    assignin('base', 'FlagQE', FlagQE);
    [NumOfRow,NumOfCol] = size(X);
    SigmaLocal = 2;
    %% =======================================================
    % normalize dataset to zero mean and unit variance
    tic;
    NormalizeCoeff1 = zeros(2,NumOfCol);
    for r=1:NumOfCol
        NormalizeCoeff1(1,r) = mean(X(:,r)); NormalizeCoeff1(2,r) = std(X(:,r));
        % zero-mean by removing the average and unit variance by dividing by the standard deviation
        X(:,r) = (X(:,r)-mean(X(:,r))) / std(X(:,r));
    end
    SOMFirstDimension1 = round(sqrt(NumOfNeurons)); SOMFirstDimension2 = round(sqrt(NumOfNeurons));
    if FlagConsole
        disp(['normalize dataset = ' num2str(toc)]);
    end
    if FlagPlot
        plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
    end
    % =======================================================
    tic;
    switch method

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
            

    end
    
    [NumOfRowNeurons,NumOfColNeurons] = size(NeuronsWeights);

    if FlagQE
        WeightRecord = evalin('base','WeightRecord');
        WeightRecord1 = WeightRecord; clear WeightRecord;
        QE = zeros(length(WeightRecord1),1);
        for r=1:length(WeightRecord1)
            BMUDistances = pdist2(WeightRecord1{r,1},X,'euclidean');
            QE(r) = mean(min(BMUDistances,[],2));
        end
        figure; plot(QE,'-','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1],'LineWidth',1.5); xlabel('Epoch'); ylabel('Quantization Error');
    end
    
    if FlagPlot
        NeuronsColorMap = parula;
        NeuronsColorMap = NeuronsColorMap(round(linspace(1,64,length(NeuronsWeights))),:);            
        figure; plot(NeuronsWeights(1,1),NeuronsWeights(1,2),'ko','MarkerFaceColor',NeuronsColorMap(1,:),'MarkerSize',7); hold on;
        for r=2:NumOfRowNeurons
            plot(NeuronsWeights(r,1),NeuronsWeights(r,2),'ko','MarkerFaceColor',NeuronsColorMap(r,:),'MarkerSize',7);
        end
        NeuronsColorMap1 = NeuronsColorMap(LabelsInitial,:);
        for r=1:NumOfRow
            plot(X(r,1),X(r,2),'o','Color',NeuronsColorMap1(r,:),'MarkerFaceColor',NeuronsColorMap1(r,:),'MarkerSize',3);
        end
        hold off; axis off;
        pbaspect([1 1 1]); daspect([1 1 1]);
    end
    
    if FlagConsole
        disp(['Training = ' num2str(toc)]);
    end
    %% =======================================================
    % denormalize dataset to original space
    tic;
    for r=1:NumOfColNeurons
        % zero-mean by removing the average and unit variance by dividing by the standard deviation
        NeuronsWeights(:,r) = (NeuronsWeights(:,r)+NormalizeCoeff1(1,r)) * NormalizeCoeff1(2,r);
        X(:,r) = (X(:,r)+NormalizeCoeff1(1,r)) * NormalizeCoeff1(2,r);
    end
    if FlagPlot
        figure; PlotResults([], NeuronsWeights, NeuronsDistances);
    end
    if FlagConsole
        disp(['denormalize dataset = ' num2str(toc)]);
    end    
    %% =======================================================
    % build the affinity matrix    
    AffMat = zeros(NumOfRowNeurons,NumOfRowNeurons);
    switch AffMatMethod
        case 'SigmaLocal'
            % as per Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering.", 2005.
            % the fowlloing distance should be the distance between two samples not squared
            tic;
            [D,~] = pdist2(NeuronsWeights,NeuronsWeights,'euclidean','Smallest',SigmaLocal);
            if FlagConsole
                disp(['build SigmaLocalMatrix = ' num2str(toc)]);
            end
            D=D';
            SigmaLocalMatrix = D(:,SigmaLocal);
            
            BytesAllocate = whos('SigmaLocalMatrix');
            if FlagConsole
                disp(['Space allocated for SigmaLocalMatrix = ' num2str(BytesAllocate.bytes)]);
            end
            assignin('base', 'BytesAllocate', BytesAllocate.bytes);

            NeuronsDistancesSparse = sparse(NeuronsDistances);
            [i,j,~] = find(NeuronsDistancesSparse);
            for r=1:length(i)
                % as per Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering.", 2005.
                % the fowlloing distance should be the square distance between two samples
                NeuronsWeightsDifference = -1 * pdist2(NeuronsWeights(i(r),:),NeuronsWeights(j(r),:),'squaredeuclidean');
                AffMat(i(r),j(r)) = NeuronsWeightsDifference / (SigmaLocalMatrix(i(r),1) * SigmaLocalMatrix(j(r),1));
                AffMat(i(r),j(r)) = exp(AffMat(i(r),j(r)));
            end
            
        case 'CONN'
            tic;
            [~,NeuronsShared] = pdist2(NeuronsWeights,X,'euclidean','Smallest',2);
            if FlagConsole
                disp(['build NeuronsShared = ' num2str(toc)]);
            end
            NeuronsShared = NeuronsShared';
            NeuronsShared = [NeuronsShared ones(NumOfRow,1)];
            NeuronsSharedMatrix = sparse(NeuronsShared(:,1),NeuronsShared(:,2),NeuronsShared(:,3),NumOfRowNeurons,NumOfRowNeurons);
            BytesAllocate = whos('NeuronsSharedMatrix');
            if FlagConsole
                disp(['Space allocated for NeuronsSharedMatrix = ' num2str(BytesAllocate.bytes)]);
            end
            assignin('base', 'BytesAllocate', BytesAllocate.bytes);
            [i,j,~] = find(NeuronsSharedMatrix);
            for r=1:length(i)
                AffMat(i(r),j(r)) = NeuronsSharedMatrix(i(r),j(r)) + NeuronsSharedMatrix(j(r),i(r));
            end
    end
    
    if FlagPlot
        figure; PlotResults([], NeuronsWeights, (rescale(AffMat,1,5) .* double(AffMat > 0)));
    end
    AffMatNNZ = round(nnz(AffMat)/2);
    if FlagConsole
        disp(['Number of edges in AffMat = ' num2str(AffMatNNZ)]);
    end
    assignin('base', 'AffMatNNZ', AffMatNNZ);
    %% =======================================================
    % The steps for spectral clustering are taken from:
    % Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems. 2002.    
    % 1 - Form the affinity matrix A
    % 2 - Define D the diagonal matrix
    tic;
    N = size(AffMat,1);
    DN = diag( 1./sqrt(sum(AffMat)+eps) );
    
    % 3 - Find k largest eigenvectors of Laplacian L to form a matrix X
    LapN = speye(N) - DN * AffMat * DN;
    [vN,D]= eig(LapN);
    % a work around if the graph Laplacian is not symmetric and the eigen
    % solver produces complex numbers, trying to symmetrize the Laplacian is not producing good results.
    if ~isreal(D)
        disp('The eigen solver produces complex numbers because the graph Laplacian is not symmetric, only the real part will be considered');
        D = real(D);
        vN = real(vN);
    end
    lambda=diag(D);
    [ls, is] = sort(lambda,'ascend');    
    if FlagPlot
        [~, isEvec2] = sort(vN(:,is(2)),'ascend');
        figure; imagesc(AffMat(isEvec2,isEvec2)); colormap(flipud(gray)); axis off;
    end
    
    % a plot to illustrate the eigengap see:
    % - Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.
    % - https://math.stackexchange.com/questions/1248131/unequal-numbers-of-eigenvalues-and-eigenvectors-in-svd
    if FlagPlot
        figure; plot(ls(1:end),'o','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1]); %ylim([0 ls(end)]);
        set(gca,'xtick',[],'ytick',[]); xlabel('eigenvectors','FontSize',18); ylabel('\lambda','FontSize',18);
    end
    
    if k~=0
        evecSelected = 2:((2+k)-2);
    else
        switch CostMethod
            case 'CostEigenGap'
                evecSelected = CostEigenGap(vN,ls,is);
            case 'CostZelnik'
                evecSelected = CostZelnik(vN,ls,is);
            case 'CostDBIOverLambda'
                evecSelected = CostDBIOverLambda(vN,ls,is);
            case 'CostDBIOverLambdaPCA'
                evecSelected = CostDBIOverLambdaPCA(vN,ls,is);
            otherwise
                disp('unable to find (evecSelected) function');
        end
    end
    
    if isempty(evecSelected) %|| length(evecSelected)==1
        disp('evecSelected is empty it was set to the second eigenvector');
        evecSelected = [2];
    end
    
    kerN = vN(:,is(evecSelected));
    
    % 4 - Form the matrix Y by normalizing X
    normN = sum(kerN .^2, 2) .^.5;
    kerNS = bsxfun(@rdivide, kerN, normN + eps);
    
    if FlagConsole
        disp(['Spectral clustering = ' num2str(toc)]);
    end
    %% =======================================================
    % setting k for kmeans
    tic;
    if k~=0
        EvalOptimalK = k;
    else
        E = evalclusters(kerNS,'kmeans','DaviesBouldin','klist',[1:NumOfNeurons]);
        EvalCriterionValues = E.CriterionValues(2:end);
        EvalEigValSum = zeros(size(EvalCriterionValues));
        for r=1:size(EvalEigValSum,2)
            EvalEigValSum(r) = sum(ls(1:r+1));
        end
        ProposedCriterion = EvalCriterionValues+EvalEigValSum;
        [~,EvalOptimalK] = min(ProposedCriterion);
        EvalOptimalK = EvalOptimalK+1;
        if FlagPlot    
            figure; hold on;
            plot(2:NumOfNeurons,ProposedCriterion(1:NumOfNeurons-1),'o-','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1],'LineWidth',1.5);
            xlabel('Number of Clusters');
        end
    end
    %% =======================================================
    % kmeans clustering
    disp('EvalOptimalK = '); disp(EvalOptimalK);
    idx = kmeans(kerNS,EvalOptimalK,'maxiter',200,'replicates',5,'EmptyAction','singleton');
    if FlagPlot
        ClusterColorMap = parula;
        ClusterColorMap = ClusterColorMap(round(linspace(1,64,length(unique(idx)))),:);
        ClusterColorMap1 = ClusterColorMap(idx,:);
        figure; plot(NeuronsWeights(1,1),NeuronsWeights(1,2),'ko','MarkerFaceColor',ClusterColorMap1(1,:),'MarkerSize',7); hold on;
        for r=2:NumOfRowNeurons
            plot(NeuronsWeights(r,1),NeuronsWeights(r,2),'ko','MarkerFaceColor',ClusterColorMap1(r,:),'MarkerSize',7);
        end
        ClusterColorMap2 = ClusterColorMap1(LabelsInitial,:);
        for r=1:NumOfRow
            plot(X(r,1),X(r,2),'o','Color',ClusterColorMap2(r,:),'MarkerFaceColor',ClusterColorMap2(r,:),'MarkerSize',3);
        end
        hold off; axis off;
        pbaspect([1 1 1]); daspect([1 1 1]);
    end
    LabelsFinal = idx(LabelsInitial);
    assignin('base', 'LabelsFinal', LabelsFinal);
    
    if FlagConsole
        disp(['kmeans clustering = ' num2str(toc)]);
    end
end