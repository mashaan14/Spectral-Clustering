function RUN_Points_Fast(X, k, FlagPlot, FlagPlotResult, FlagNormalize)
    
    assignin('base', 'FlagPlot', FlagPlot);        
%     XFull = X;
%     [X,ia,ic] = unique(X,'rows');
    [NumOfRow,NumOfCol] = size(X);
    MaxK = min(NumOfRow,20);
    % normalize dataset to zero mean and unit variance
    if FlagNormalize        
        NormalizeCoeff1 = zeros(2,NumOfCol);
        for r=1:NumOfCol
            NormalizeCoeff1(1,r) = mean(X(:,r)); NormalizeCoeff1(2,r) = std(X(:,r));
            % zero-mean by removing the average and unit variance by dividing by the standard deviation
            X(:,r) = (X(:,r)-mean(X(:,r))) / std(X(:,r));
        end
    end
    
    %% distance matrix for n points
    % compute distance to k^th nieghbour
    LocalSigmaK = min(NumOfRow-2,1000);
    disp(['NumOfRow = ' num2str(NumOfRow)]);
    disp(['LocalSigmaK = ' num2str(LocalSigmaK)]);
    % as per Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering.", 2005.
    % the fowlloing distance should be the distance between two samples not squared
    [LocalSigmaD,LocalSigmaI] = pdist2(X,X,'euclidean','Smallest',LocalSigmaK+1);
    LocalSigmaD = LocalSigmaD';
    LocalSigmaI = LocalSigmaI';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % remove first column since it indicates distance to the point itself
    ALocalSigmaD = LocalSigmaD(:,2:end);
    [LocalSigmaDRow,LocalSigmaDCol] = size(ALocalSigmaD);
    
    %% filtering the graph      
    % Create a histogram of all distances in the dataset where bin edges are determined via FD rule.
    % These bin edges will create a common ground to evaluate each point separately.
    [~,LocalSigmaDHistEdges] = histcounts(LocalSigmaD(:,2:end),'BinMethod','fd');    
    LocalSigmaDHistCenters = LocalSigmaDHistEdges(1:end-1) + diff(LocalSigmaDHistEdges) / 2;
    LocalSigmaDHistMat = zeros(NumOfRow,length(LocalSigmaDHistEdges)-1);
    for r=1:NumOfRow
        % a histogram for each point
        [LocalSigmaDHistMat(r,:),~] = histcounts(LocalSigmaD(r,2:end),LocalSigmaDHistEdges);        
    end
    BinCountsLength = size(LocalSigmaDHistMat,2);
    
    % Moving weighted average MWA smoothing
    % shift signal hrizontaly to add elements vertically for window size 3 of MWA
    
    % original signal
    s1 = LocalSigmaDHistMat;
    % original signal shifted by 1
    s2 = [zeros(NumOfRow,1) s1(:,1:end-1)];
    % original signal shifted by 2
    s3 = [zeros(NumOfRow,2) s1(:,1:end-2)];
    % original signal ranks
    si1 = repmat(1:size(s1,2),NumOfRow,1);
    % original signal ranks shifted by 1
    si2 = [zeros(NumOfRow,1) si1(:,1:end-1)];
    % original signal ranks shifted by 2
    si3 = [zeros(NumOfRow,2) si1(:,1:end-2)];
    % Multiply elements in each window of size 3
    S=s1 .* s2 .* s3;
    SI=si1 .* si2 .* si3;
    % signal is weighted by elements ranks such that elements with lower ranks would get a higher weight
    % the intuition behind it, we're looking for the maximum peak in a histogram, the peaks with lower distances are more important
    SS = S ./ SI;
    % execlude first two elements because MWA window is 3
    SS1 = SS(:,3:end);
    % replicate the mean of each row to match the number of columns for fast subtraction
    SS1Mean = repmat(mean(SS1,2),1,BinCountsLength-2);
    % subtract the mean of each row from all columns
    SS2 = SS1 - SS1Mean;
    % find the first positive element (i.e., first element greater than the mean)
    [~,SSI]= max(SS2>0,[],2);
%     [~,SSI]= max(SS2,[],2);
    % because we execluded the first two columns we have to add 2 to get the right index
    SSI = SSI + 2;
    % because we want the right edge of the bin we have to add 1
    SSI = SSI + 1;
    % retrieve the bin edge corresponding with maximum element
    LocalSigmaK7 = LocalSigmaDHistEdges(SSI)';
    
    clearvars -except X ia ic k CostMethod NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK LocalSigmaD LocalSigmaI LocalSigmaK LocalSigmaK7
    
    LocalSigmaD1 = LocalSigmaD;
    % Histogram threshold
    % any distance that is larger than the threshold set by each point, would be set to zero
    for r=1:NumOfRow
        LocalSigmaD1(r,LocalSigmaD1(r,:)>LocalSigmaK7(r)) = 0;
    end

    % store the filtered distance matrix into a new variable
    LocalSigmaDFiltered1 = LocalSigmaD1;
    % convert zeros (i.e., removed distances) into NaN values for to be execluded from mean and std computations
    LocalSigmaDFiltered1(LocalSigmaDFiltered1==0) = NaN;
    % compute the mean with zeros execluded
    LocalSigmaDMean = nanmean(LocalSigmaDFiltered1,2);
    
    % create a matrix where each element [i,j] = local \sigma [i] * local \sigma [j]
    LocalSigmaDMeanTemp1 = repmat(LocalSigmaDMean,1,LocalSigmaK+1);
    LocalSigmaDMeanTemp2 = LocalSigmaDMean(LocalSigmaI);
    LocalSigmaDMean1 = LocalSigmaDMeanTemp1 .* LocalSigmaDMeanTemp2;
    % store the filtered distance matrix into a new variable
    LocalSigmaD2 = LocalSigmaD;
%     LocalSigmaD2(isnan(LocalSigmaD2)) = 0;
    LocalSigmaD2 = LocalSigmaD2 .^ 2;
    LocalSigmaD2 = LocalSigmaD2 .* -1;
    LocalSigmaD2 = LocalSigmaD2 ./ LocalSigmaDMean1(:,1:LocalSigmaK+1);
    LocalSigmaD2 = exp(LocalSigmaD2);
    LocalSigmaD2 = LocalSigmaD2 .* double(LocalSigmaD~=0);
    
    % get bin edges for the histogram of distances penalized by local sigma
    [~,LocalSigmaD2HistEdges] = histcounts(LocalSigmaD2(:,2:end),'BinMethod','fd');
    % nullify entries in the far left bin because they are very small
    LocalSigmaD2(LocalSigmaD2<=LocalSigmaD2HistEdges(2)) = 0;
    
    % store the filtered distance matrix into a new variable
    LocalSigmaDFiltered2 = LocalSigmaD2;
    % convert zeros (i.e., removed distances) into NaN values for to be execluded from mean and std computations
    LocalSigmaDFiltered2(LocalSigmaDFiltered2==0) = NaN;
    % compute mean with zeros execluded
    LocalSigmaDMean2 = nanmean(LocalSigmaDFiltered2,2);
    % compute std with zeros execluded
    LocalSigmaDStd2 = nanstd(LocalSigmaDFiltered2,[],2);
%     if FlagPlot
%         figure; histogram(LocalSigmaD2,'BinMethod','fd');
%         figure; histogram(LocalSigmaDFiltered2,'BinMethod','fd');
%         
%         pDense = 580;
%         figure; hold on;
%         histogram(LocalSigmaDFiltered2(pDense,:),'BinMethod','fd');
%         line([LocalSigmaDMean2(pDense)-LocalSigmaDStd2(pDense) LocalSigmaDMean2(pDense)-LocalSigmaDStd2(pDense)], ylim, 'Color','r','LineStyle','--');
%         line([LocalSigmaDMean2(pDense) LocalSigmaDMean2(pDense)], ylim, 'Color','r');
%         line([LocalSigmaDMean2(pDense)+LocalSigmaDStd2(pDense) LocalSigmaDMean2(pDense)+LocalSigmaDStd2(pDense)], ylim, 'Color','r','LineStyle','--');
%         
%         pSparse = 64;
%         figure; hold on;
%         histogram(LocalSigmaDFiltered2(pSparse,:),'BinMethod','fd');
%         line([LocalSigmaDMean2(pSparse)-LocalSigmaDStd2(pSparse) LocalSigmaDMean2(pSparse)-LocalSigmaDStd2(pSparse)], ylim, 'Color','r','LineStyle','--');
%         line([LocalSigmaDMean2(pSparse) LocalSigmaDMean2(pSparse)], ylim, 'Color','r');
%         line([LocalSigmaDMean2(pSparse)+LocalSigmaDStd2(pSparse) LocalSigmaDMean2(pSparse)+LocalSigmaDStd2(pSparse)], ylim, 'Color','r','LineStyle','--');
%     end
    
    for r=1:NumOfRow
        if (LocalSigmaDMean2(r)+LocalSigmaDStd2(r)) >= max(LocalSigmaD2(r,:))
            LocalSigmaD2(r,LocalSigmaD2(r,:) < (LocalSigmaDMean2(r)-LocalSigmaDStd2(r))) = 0;
        else
            LocalSigmaD2(r,LocalSigmaD2(r,:) < (LocalSigmaDMean2(r)+LocalSigmaDStd2(r))) = 0;
        end
    end

    % retrieve nieghbours indices to create edges list
    LocalSigmaI1 = LocalSigmaI(:,1);
    LocalSigmaI1 = repmat(LocalSigmaI1,1,LocalSigmaK);
    LocalSigmaI1 = reshape(LocalSigmaI1,[],1);
    LocalSigmaI2 = LocalSigmaI(:,2:end);
    LocalSigmaI2 = reshape(LocalSigmaI2,[],1);
    LocalSigmaI3 = [LocalSigmaI1 LocalSigmaI2];
%     LocalSigmaD1 = [LocalSigmaDMean(LocalSigmaI1) LocalSigmaDMean(LocalSigmaI2)];
    
    LocalSigmaD2 = LocalSigmaD2(:,2:end);
    LocalSigmaD2 = reshape(LocalSigmaD2,[],1);
    % as per Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering.", 2005.
    % the distance should be the square distance between two samples
%     LocalSigmaD2 = LocalSigmaD2 .^ 2;
%     LocalSigmaD2 = LocalSigmaD2 .* -1;
%     LocalSigmaD3 = LocalSigmaD1(:,1) .* LocalSigmaD1(:,2);
%     LocalSigmaD3 = LocalSigmaD3 + eps;
%     LocalSigmaD4 = LocalSigmaD2 ./ LocalSigmaD3;
%     LocalSigmaD4 = exp(LocalSigmaD4);
%     LocalSigmaD4 = LocalSigmaD4 .* double(LocalSigmaD2~=0);
%     Y6 = [LocalSigmaI1 LocalSigmaI2 LocalSigmaD4];
    Y6 = [LocalSigmaI1 LocalSigmaI2 LocalSigmaD2];
    Y6(Y6(:,3)==0,:) = [];
    LocalSigmaGraph = graph(Y6(:,1),Y6(:,2),Y6(:,3));
    EdgesNumMutual = numedges(LocalSigmaGraph);
    assignin('base', 'edgesNumMutual', EdgesNumMutual);
    
    clearvars -except X ia ic k CostMethod NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK LocalSigmaGraph
    
    %% keep edges with mutual agreement
    % retrieve edges
    LocalSigmaGraphTemp1 = table2array(LocalSigmaGraph.Edges);
    % set the wieght for each edge to one, for easier counting
    LocalSigmaGraphTemp1 = [LocalSigmaGraphTemp1 ones(length(LocalSigmaGraphTemp1),1)];
    % create a graph with multiple edges
    LocalSigmaGraphTemp = graph(LocalSigmaGraphTemp1(:,1),LocalSigmaGraphTemp1(:,2),LocalSigmaGraphTemp1(:,4));
    % replace multiple edges with one edge carrying their sum, this should give us the number of edges
    LocalSigmaGraphTemp = simplify(LocalSigmaGraphTemp,'sum');
    % replace multiple edges with one edge carrying their sum, this should give us the mean weight on edges
    LocalSigmaGraph = simplify(LocalSigmaGraph,'mean');
    % put edges lists next to eachother
    LocalSigmaGraphTemp2 = [table2array(LocalSigmaGraph.Edges) table2array(LocalSigmaGraphTemp.Edges)];
    % keep edges that have a count of more than two
    LocalSigmaGraphTemp3 = LocalSigmaGraphTemp2(LocalSigmaGraphTemp2(:,6)>=2,:);
    LocalSigmaGraph = graph(LocalSigmaGraphTemp3(:,1),LocalSigmaGraphTemp3(:,2),LocalSigmaGraphTemp3(:,3),NumOfRow);
    
    EdgesPercent = (numedges(LocalSigmaGraph) / (NumOfRow*(NumOfRow-1)/2)) * 100;
    assignin('base', 'edgesPercent', EdgesPercent);
    assignin('base', 'edgesNum', numedges(LocalSigmaGraph));
            
    % draw the corresponding graph
    GraphEdgeWidth = 5;
    LWidths = GraphEdgeWidth*LocalSigmaGraph.Edges.Weight/max(LocalSigmaGraph.Edges.Weight);
    if FlagPlot
        figure; hold on;
        plot(LocalSigmaGraph,'XData',X(:,1),'YData',X(:,2),...
            'EdgeLabel',[],'LineWidth',LWidths,...
            'NodeLabel',[],'MarkerSize',3,'NodeColor',[0.5 0.5 1],...
            'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2],'EdgeFontSize',18);
        hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
    end

    clearvars -except X ia ic k CostMethod NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK LocalSigmaGraph SumEdgeTrackKeep L LabelsInitial XFilt
   
    %% Spectral clustering
    % The steps for spectral clustering are taken from:
    % Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems. 2002.    
    % 1 - Form the affinity matrix A
    % 2 - Define D the diagonal matrix
    LocalSigmaGraphEdgesHalf = table2array(LocalSigmaGraph.Edges);
    LocalSigmaGraphEdges = [LocalSigmaGraphEdgesHalf; [LocalSigmaGraphEdgesHalf(:,2) LocalSigmaGraphEdgesHalf(:,1) LocalSigmaGraphEdgesHalf(:,3)]];    
    AffMat = sparse(LocalSigmaGraphEdges(:,1),LocalSigmaGraphEdges(:,2),LocalSigmaGraphEdges(:,3),NumOfRow,NumOfRow);
    clear LocalSigmaGraph LocalSigmaGraphEdges
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
%     [vN,D]= eigs(LapN,min(NumOfRow,100),eps);
    [~,D,vN] = svds(LapN,min(NumOfRow,30),'smallest');
    clearvars -except X ia ic k CostMethod NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK SumEdgeTrackKeep L LabelsInitial XFilt LocalSigmaGraphEdgesHalf vN D
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
                if sum(ls==0) > 1
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
        plot(ls(1:20),'o','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1]); %ylim([0 ls(end)]);
        plot(k,ls(k),'o','color',[0.6350, 0.0780, 0.1840],'MarkerFaceColor',[0.6350, 0.0780, 0.1840],'MarkerSize',10);
        xlabel('eigenvectors','FontSize',18); ylabel('\lambda','FontSize',18); % set(gca,'xtick',[],'ytick',[]);
    end               
    kerN = vNSort;
    
    % 4 - Form the matrix Y by normalizing X
    normN = sum(kerN .^2, 2) .^.5;
    kerNS = bsxfun(@rdivide, kerN, normN + eps);
    
    clearvars -except X ia ic k CostMethod NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK SumEdgeTrackKeep L LabelsInitial XFilt LocalSigmaGraphEdgesHalf kerNS ls
    
    LabelsDecimal = zeros(size(kerNS,1),k);
    for r=2:k
        LabelsDecimal(:,r) = kmeans(kerNS(:,2:r),k,'maxiter',500,'replicates',3,'EmptyAction','singleton');
    end
    
    % IntraClusterWeights is the sum of all edges where vertices are in the same class
    % InterClusterWeights is the sum of all edges where vertices are in the different classes
    % CoherenceIndex for a certain eigenvector = IntraClusterWeights / InterClusterWeights
    CoherenceIndex = zeros(1,k);
    for r=2:k
        LocalSigmaGraph2Edges1 = LocalSigmaGraphEdgesHalf;
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
    clearvars -except X ia ic k CostMethod NumOfRow NumOfCol FlagPlot FlagPlotResult MaxK SumEdgeTrackKeep L LabelsInitial XFilt LabelsBest
        
%     LabelsFinal = LabelsBest(ic);
    LabelsFinal = LabelsBest;
    if FlagPlotResult
        ClusterColorMap = parula;
        ClusterColorMap = ClusterColorMap(round(linspace(1,64,length(unique(LabelsBest)))),:);
        figure; hold on;
        for r=1:size(ClusterColorMap,1)
            plot(X(LabelsBest==r,1),X(LabelsBest==r,2),'o','Color',ClusterColorMap(r,:),'MarkerFaceColor',ClusterColorMap(r,:),'MarkerSize',5);
        end
        hold off; axis off;
        pbaspect([1 1 1]); daspect([1 1 1]);
    end
    assignin('base', 'LabelsFinal', LabelsFinal);

end