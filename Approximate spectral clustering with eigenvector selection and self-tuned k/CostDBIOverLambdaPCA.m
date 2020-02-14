function [evecSelected] = CostDBIOverLambdaPCA(vN,ls,is)
    %% compute eigenvectors cost
    FlagPlot = evalin('base', 'FlagPlot');
%     evecNum = round(0.5 * length(ls));

    evecNum = length(ls);
    evecCost = zeros(evecNum,1);

    figure;
    for r=1:evecNum
        evecCurrent = vN(:,is(r));
        E = evalclusters(evecCurrent,'kmeans','DaviesBouldin','klist',1:5);

        evecCost(r) = (E.CriterionValues(2)+E.CriterionValues(3)+E.CriterionValues(4)) / ls(r);
    end
    assignin('base', 'evecCost', evecCost);
    evecCost1 = evecCost(2:end);    
    %% create a hstogram of eigenvectors cost
    % create a histogram plot using The Freedman-Diaconis rule, which is less sensitive to
    % outliers in the data, and might be more suitable for data with heavy-tailed distributions. 
    % It uses a bin width of 2*IQR(X(:))*numel(X)^(-1/3), where IQR is the interquartile range of X.
    % Original Paper:
    %   Freedman, David, and Persi Diaconis. "On the histogram as a density estimator: L 2 theory." 
    %   Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete 57.4 (1981): 453-476.
    figure;
    evecCostHist = histogram(evecCost1,'BinMethod','fd');
    % retrieve the frequency for each bin
    evecCostFrequency = evecCostHist.BinCounts;
    % in the edge vector edges(1) is the left edge of the first bin, and edges(end) is the right edge of the last bin.
    evecCostHistEdges = evecCostHist.BinEdges;
    % find the mid point between consecutive edges to be the bin center
    evecCostHistCenters = evecCostHistEdges(1:end-1) + diff(evecCostHistEdges) / 2;
    % calculate the mean of the frequency table (histogram) as explained in the following link:
    %   https://www.mathsisfun.com/data/mean-frequency-table.html
    evecCostHistMean = sum(evecCostFrequency .* evecCostHistCenters) / sum(evecCostFrequency);
    % claculate the variance and standard deviation of the frequency table (histogram) as explained in the following link:
    %   http://www.statcan.gc.ca/edu/power-pouvoir/ch12/5214891-eng.htm
    evecCostVariance = sum(evecCostFrequency .* ((evecCostHistCenters - evecCostHistMean) .^ 2)) / sum(evecCostFrequency);
    evecCostHistStd = sqrt(evecCostVariance);
%     evecCostHistStd = 2*evecCostHistStd;

    if FlagPlot
        figure; histogram(evecCost1,'BinMethod','fd'); hold on;
        line([evecCostHistMean-evecCostHistStd evecCostHistMean-evecCostHistStd], ylim, 'Color','r');
        line([evecCostHistMean+evecCostHistStd evecCostHistMean+evecCostHistStd], ylim, 'Color','r');
        set(gca,'xtick',[],'ytick',[]); xlabel('Relevance Metric','FontSize',18); ylabel('Number of eigenvectors','FontSize',18);
        hold off;
    end
    
    evecCost2 = evecCost1;
    evecCost2(evecCost2>=evecCostHistMean-evecCostHistStd&evecCost2<=evecCostHistMean+evecCostHistStd) = 0;
    evecSelected = 1:(find(evecCost2==0,1)-1);
    if isempty(evecSelected) %|| length(evecSelected)==1
        disp('evecSelected is empty it was set to the second eigenvector');
        evecSelected = [1];
    end
    evecSelected = evecSelected + 1;
    disp('evecSelected = '); disp(evecSelected);
%% filter based on PCA explained variance
    evecSelectedFiltPCA=evecSelected;
    if length(evecSelected) > 2
        [coeff, score, latent, tsquared, explained, mu] = pca(vN(:,is(evecSelected)));
        cumexplained = cumsum(explained);
    %     cumunexplained = 100 - cumexplained;
        figure; plot(1:length(explained), cumexplained,'o-','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1]);
        ylim([0 100]); set(gca, 'xtick', [2:1:length(evecSelected)+2]);
        xlabel('Eigenvectors','FontSize',18); ylabel('Explained variance','FontSize',18)

        [VarExplainFilt,~,~] = find(cumexplained>80);    
        evecSelectedFiltPCA = evecSelectedFiltPCA(1:VarExplainFilt(1)-1);
    end
    if isempty(evecSelectedFiltPCA) %|| length(evecSelected)==1
        disp('evecSelectedFiltPCA is empty it was set to the second eigenvector');
        evecSelectedFiltPCA = [2];
    end
    if length(evecSelectedFiltPCA) < 2
        evecSelectedFiltPCA = [evecSelectedFiltPCA evecSelectedFiltPCA(end)+1];
    end
    disp('evecSelectedFiltPCA = '); disp(evecSelectedFiltPCA);
    evecSelected=evecSelectedFiltPCA;
    %% start plotting
    if FlagPlot
        figure;
        for r=1:9
%             subplot(3,3,r);
            plot(1:size(vN,1),sort(transpose(vN(:,r)),2,'ascend'),'o','color',[0.3 0.3 1],'MarkerFaceColor',[0.3 0.3 1]);
            PlotTitle = ['Neurons positions based on eigenvector ' num2str(r)];
            ylim([-1 1]);
%             title(PlotTitle);
        end
    end
end