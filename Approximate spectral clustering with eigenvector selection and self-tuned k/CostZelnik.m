function [evecSelected] = CostZelnik(vN,ls,is)
    FlagPlot = evalin('base', 'FlagPlot');
    if length(ls)>20
        evecNum = 20;
    else
        evecNum = length(ls);
    end
    clusts = {1,evecNum}; Vr = {1,evecNum}; Quality = Inf*ones(1,evecNum);
    
    % sort based on eigenvalues
    vN = vN(:,is);
    %% embedding space rotation
    for r=1:evecNum        
        % Form an embedding space and rotate it according to Zelnik-Manor and Perona method
        SpaceCurr = vN(:,1:r);
        [clusts{r},Quality(r),Vr{r}] = evrot(SpaceCurr,1);        
    end
    if FlagPlot    
        figure; plot(1:evecNum,Quality,'o-','color',[1 0.3 0.3],'MarkerFaceColor',[1 0.3 0.3],'LineWidth',1.5);        
        xlabel('Number of Dimensions'); legend('Quality');
    end
    %% selecting the number of dimensions that yields the lowes rotation cost
    [~,QualityBest] = min(Quality);
    evecSelectedFiltZelnik = 1:QualityBest;
    evecSelectedFiltZelnik(1) = [];
    disp('evecSelectedFiltZelnik = '); disp(evecSelectedFiltZelnik);
    evecSelected=evecSelectedFiltZelnik;
end