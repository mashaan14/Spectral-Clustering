function RUN_Points(X, method, NumOfNeurons, FlagPlot, FlagQE, FlagConsole, AffMatMethod, CostMethod, k)

    assignin('base', 'FlagPlot', FlagPlot);
    assignin('base', 'FlagQE', FlagQE);
    [NumOfRow,NumOfCol] = size(X);
    SigmaLocal = 2;
    if FlagPlot
        plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
        axis off; pbaspect([1 1 1]); daspect([1 1 1]);
    end
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
    % =======================================================
    TimeToTrainStart = tic;
    switch method
               
        case 'kmeans+CHL'
            FlagQE = false;
            
            [LabelsInitial,NeuronsWeights,~,DistToNeurons] = kmeans(X,NumOfNeurons,'maxiter',500,'replicates',3,'EmptyAction','singleton');
            [NumOfRowNeurons,NumOfColNeurons] = size(NeuronsWeights);
            % denormalize dataset to original space
            for r=1:NumOfColNeurons
                % zero-mean by removing the average and unit variance by dividing by the standard deviation
                NeuronsWeights(:,r) = (NeuronsWeights(:,r)+NormalizeCoeff1(1,r)) * NormalizeCoeff1(2,r);
                X(:,r) = (X(:,r)+NormalizeCoeff1(1,r)) * NormalizeCoeff1(2,r);
            end
            %% CONN matrix
            [DistToNeuronsSort,DistToNeuronsSortIndex] = sort(DistToNeurons,2);
            DistToNeuronsSortIndex = DistToNeuronsSortIndex(:,1:2);
            [NeuronsUniqBMU,~,NeuronsUniqBMUOccur] = unique(DistToNeuronsSortIndex,'rows');
            NeuronsUniqBMUOccur1 = accumarray(NeuronsUniqBMUOccur, 1);
            NeuronsUniqBMU1 = [NeuronsUniqBMU NeuronsUniqBMUOccur1];
            NeuronsUniqBMU2 = sparse(NeuronsUniqBMU1(:,1),NeuronsUniqBMU1(:,2),NeuronsUniqBMU1(:,3),NumOfNeurons,NumOfNeurons);
            NeuronsDistancesCONN = zeros(NumOfNeurons,NumOfNeurons);
            NeuronsDistancesCONNMutual = zeros(NumOfNeurons,NumOfNeurons);
            [i,j,~] = find(NeuronsUniqBMU2);
            for r=1:length(i)
                NeuronsDistancesCONN(i(r),j(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                NeuronsDistancesCONN(j(r),i(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                if ~isempty(find(NeuronsUniqBMU2(i(r),j(r)),1)) && ~isempty(find(NeuronsUniqBMU2(j(r),i(r)),1))
                    NeuronsDistancesCONNMutual(i(r),j(r)) = NeuronsUniqBMU2(i(r),j(r));
                    NeuronsDistancesCONNMutual(j(r),i(r)) = NeuronsUniqBMU2(j(r),i(r));
                end
            end
            NeuronsGraphCONN = graph(NeuronsDistancesCONN);
            GraphEdgeWidth = 5;
            LWidths = GraphEdgeWidth*NeuronsGraphCONN.Edges.Weight/max(NeuronsGraphCONN.Edges.Weight);
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphCONN,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',[],'LineWidth',LWidths,...
                    'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                %NeuronsGraphCONN.Edges.Weight
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
            
           %% Directed CONN Graph
           NeuronsGraphCONNDir = digraph(NeuronsUniqBMU1(:,1),NeuronsUniqBMU1(:,2),NeuronsUniqBMU1(:,3));
           LWidths = GraphEdgeWidth*NeuronsGraphCONNDir.Edges.Weight/max(NeuronsGraphCONNDir.Edges.Weight);
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphCONNDir,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',[],'LineWidth',LWidths,'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2],'ArrowSize',10);
                % NeuronsGraphCONNDir.Edges.Weight
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
            
            NeuronsGraphCONNDirMutual = digraph(NeuronsDistancesCONNMutual);
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphCONNDirMutual,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',[],'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                %NeuronsGraphCONNDirMutual.Edges.Weight
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
            DirectWeight = table2array(NeuronsGraphCONNDirMutual.Edges);
            DirectWeightMat = sparse(DirectWeight(:,1),DirectWeight(:,2),DirectWeight(:,3),NumOfNeurons,NumOfNeurons);
            DirectWeightMat = full(DirectWeightMat);
            DirectWeightDiff = DirectWeightMat - transpose(DirectWeightMat);
            DirectWeightDiff = abs(DirectWeightDiff);
            EdgeBalance = mean(nonzeros(triu(DirectWeightDiff,1)));
            EdgeBalancePlus = EdgeBalance + std(nonzeros(triu(DirectWeightDiff,1)));
            
            %% CONNFilt matrix
            NeuronsDistancesCONNFilt = zeros(NumOfNeurons,NumOfNeurons);            
            for r=1:length(i)
                if ~isempty(find(NeuronsUniqBMU2(i(r),j(r)),1)) && ~isempty(find(NeuronsUniqBMU2(j(r),i(r)),1))
                    NeuronsDistancesCONNFilt(i(r),j(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                    NeuronsDistancesCONNFilt(j(r),i(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                elseif ~isempty(find(NeuronsUniqBMU2(i(r),j(r)),1)) && isempty(find(NeuronsUniqBMU2(j(r),i(r)),1))
                    if NeuronsUniqBMU2(i(r),j(r)) <= EdgeBalancePlus
                        NeuronsDistancesCONNFilt(i(r),j(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                        NeuronsDistancesCONNFilt(j(r),i(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                    end
                elseif isempty(find(NeuronsUniqBMU2(i(r),j(r)),1)) && ~isempty(find(NeuronsUniqBMU2(j(r),i(r)),1))
                    if NeuronsUniqBMU2(j(r),i(r)) <= EdgeBalancePlus
                        NeuronsDistancesCONNFilt(i(r),j(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                        NeuronsDistancesCONNFilt(j(r),i(r)) = NeuronsUniqBMU2(i(r),j(r)) + NeuronsUniqBMU2(j(r),i(r));
                    end
                end
            end
            NeuronsGraphCONNFilt = graph(NeuronsDistancesCONNFilt);
            LWidths = GraphEdgeWidth*NeuronsGraphCONNFilt.Edges.Weight/max(NeuronsGraphCONNFilt.Edges.Weight);
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphCONNFilt,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',NeuronsGraphCONNFilt.Edges.Weight,'LineWidth',LWidths,...
                    'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
           %% voting matrix           
           NeuronsDistancesVote = zeros(NumOfNeurons,NumOfNeurons);
           NeuronsDistancesVote(NeuronsDistancesCONNFilt~=0) = 1;
%            NeuronsDistancesVote(NeuronsDistancesCONN~=0) = 1;
           
            NeuronsDistancesEuc = pdist2(NeuronsWeights,NeuronsWeights,'squaredeuclidean');
            NeuronsDistancesEuc(NeuronsDistancesCONNFilt==0) = 0;
%             NeuronsDistancesEuc(NeuronsDistancesCONN==0) = 0;
            NeuronsAcceptRange = zeros(2,NumOfNeurons);
            for r=1:NumOfNeurons
                NeuronsAcceptRange(1,r) = mean(nonzeros(NeuronsDistancesEuc(r,:)));
                NeuronsAcceptRange(2,r) =  std(nonzeros(NeuronsDistancesEuc(r,:)));
            end
            
            for r=1:NumOfNeurons
                AcceptRangeIdx = find(NeuronsDistancesEuc(r,:) >...
                    (NeuronsAcceptRange(1,r)+(1*NeuronsAcceptRange(2,r))...
                    ));
                if ~isempty(AcceptRangeIdx)
                    NeuronsDistancesVote(r,AcceptRangeIdx) = NeuronsDistancesVote(r,AcceptRangeIdx) + 1;
                    NeuronsDistancesVote(AcceptRangeIdx,r) = NeuronsDistancesVote(AcceptRangeIdx,r) + 1;
                end                
            end
            
            NeuronsGraphVote = graph(NeuronsDistancesVote);                        
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphVote,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',(NeuronsGraphVote.Edges.Weight)-1,'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                set(gca,'FontSize',18)
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
            
            %% create filtered CONN matrix
            NeuronsGraphVote1 = rmedge(NeuronsGraphVote,find((NeuronsGraphVote.Edges.Weight)-1 >= 1));
            NeuronsDistancesVote1 = full(adjacency(NeuronsGraphVote1));
            NeuronsGraphVote1 = graph(NeuronsDistancesVote1);                        
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphVote1,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',(NeuronsGraphVote1.Edges.Weight)-1,'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
            
            NeuronsDistancesCONNFilt1 = NeuronsDistancesCONNFilt;
            NeuronsDistancesCONNFilt1(NeuronsDistancesVote1==0) = 0;
            NeuronsGraphCONNFilt1 = graph(NeuronsDistancesCONNFilt1);
           LWidths = GraphEdgeWidth*NeuronsGraphCONNFilt1.Edges.Weight/max(NeuronsGraphCONNFilt1.Edges.Weight);
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphCONNFilt1,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',[],'LineWidth',LWidths,...
                    'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                % NeuronsGraphCONNFilt1.Edges.Weight
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end

            %% create filtered CONN matrix            
            NeuronsGraphVote2 = rmedge(NeuronsGraphVote,find((NeuronsGraphVote.Edges.Weight)-1 >= 2));
            NeuronsDistancesVote2 = full(adjacency(NeuronsGraphVote2));
            NeuronsGraphVote2 = graph(NeuronsDistancesVote2);                        
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphVote2,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',(NeuronsGraphVote2.Edges.Weight)-1,'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
            
            NeuronsDistancesCONNFilt2 = NeuronsDistancesCONNFilt;
            NeuronsDistancesCONNFilt2(NeuronsDistancesVote2==0) = 0;
            NeuronsGraphCONNFilt2 = graph(NeuronsDistancesCONNFilt2);
            LWidths = GraphEdgeWidth*NeuronsGraphCONNFilt2.Edges.Weight/max(NeuronsGraphCONNFilt2.Edges.Weight);
            if FlagPlot
                figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3);
                hold on;
                plot(NeuronsGraphCONNFilt2,'XData',NeuronsWeights(:,1),'YData',NeuronsWeights(:,2),...
                    'EdgeLabel',[],'LineWidth',LWidths,...
                    'NodeLabel',[],'MarkerSize',7,'NodeColor',[1 1 0.25],...
                    'EdgeAlpha',1,'EdgeColor',[1 0.2 0.2]);
                % NeuronsGraphCONNFilt2.Edges.Weight
                hold off; axis off; pbaspect([1 1 1]); daspect([1 1 1]);
            end
    end
        
    TimeToTrain = toc(TimeToTrainStart);
    if FlagConsole
        disp(['Training = ' num2str(TimeToTrain)]);
    end
    assignin('base', 'TimeToTrain', TimeToTrain);        
    
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
    %% =======================================================
    % build the affinity matrix    
    AffMat = zeros(NumOfRowNeurons,NumOfRowNeurons);
    switch AffMatMethod       
        case 'CONN'
            AffMat = NeuronsDistancesCONN;
%             NumOfEdges = numedges(NeuronsGraphCONN);
        case 'CONNFilt'
            AffMat = NeuronsDistancesCONNFilt;
%             NumOfEdges = numedges(NeuronsGraphCONNFilt);
        case 'EucInv'
            AffMat = NeuronsDistancesCONNFilt1;
%             NumOfEdges = numedges(NeuronsGraphCONNFilt1);
        case 'EucScale'
            AffMat = NeuronsDistancesCONNFilt2;
%             NumOfEdges = numedges(NeuronsGraphCONNFilt2);
        otherwise
            disp('unable to find AffMat method');
    end    
%     assignin('base', 'NumOfEdges', NumOfEdges);
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
%         evecSelected = 1:((1+k)-1);
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
%     switch CostMethod
%         case 'CostEigenGap'
%             EvalOptimalK = length(evecSelected);
%         case 'CostGMM'
%             gm={1,NumOfNeurons-1}; BIC=zeros(1,NumOfNeurons-1);
%             for k = 1:NumOfNeurons-1
%                 gm{k} = fitgmdist(kerNS,k,'RegularizationValue',0.000001);
%                 BIC(k)= gm{k}.BIC;
%             end
%             [~,EvalOptimalK] = min(BIC);
%             if FlagPlot    
%                 figure; hold on;
%                 plot(2:NumOfNeurons-1,BIC(2:NumOfNeurons-1),'o-','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1],'LineWidth',1.5);
%                 xlabel('Number of Clusters');
%             end 
%         case 'CostDBI'
%             E = evalclusters(kerNS,'kmeans','DaviesBouldin','klist',[1:NumOfNeurons]);
%             EvalCriterionValues = E.CriterionValues(2:end);            
%             [~,EvalOptimalK] = min(EvalCriterionValues);
%             EvalOptimalK = EvalOptimalK+1;
%             if FlagPlot    
%                 figure; hold on;
%                 plot(2:NumOfNeurons,E.CriterionValues(2:NumOfNeurons),'o-','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1],'LineWidth',1.5);
%                 xlabel('Number of Clusters');
%             end            
%         case 'CostDBIOverLambda'
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
%     end
    end
    %% =======================================================
    % kmeans clustering
    disp('EvalOptimalK = '); disp(EvalOptimalK);
    idx = kmeans(kerNS,EvalOptimalK,'maxiter',500,'replicates',3,'EmptyAction','singleton');
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