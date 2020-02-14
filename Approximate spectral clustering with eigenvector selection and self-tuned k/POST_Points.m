LabelsFinal1=bestMap(XLabels,LabelsFinal);
LabelsFinal1=LabelsFinal1';
hitrate = sum(XLabels(:) == LabelsFinal1(:)) / length(XLabels);
disp('hitrate = '); disp(hitrate);