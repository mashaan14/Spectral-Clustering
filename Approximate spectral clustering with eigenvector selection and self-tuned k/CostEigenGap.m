function [evecSelected] = CostEigenGap(~,ls,~)
    FlagPlot = evalin('base', 'FlagPlot');
    if length(ls)>20
        evecNum = 20;
    else
        evecNum = length(ls);
    end
    evecCost = ls;
    evecSelected = 1:evecNum;
    assignin('base', 'evecCost', evecCost);
    evecCost1 = diff(evecCost(1:evecNum));
    [~,EigenGapCount]=max(evecCost1);
    evecSelectedFiltGap = evecSelected(:,1:EigenGapCount);
    evecSelectedFiltGap(1) = [];
    disp('evecSelectedFiltGap = '); disp(evecSelectedFiltGap);
    evecSelected=evecSelectedFiltGap;
end