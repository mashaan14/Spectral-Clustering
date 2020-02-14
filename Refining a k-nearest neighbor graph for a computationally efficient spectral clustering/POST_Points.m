addpath 'C:\Users\mals6571\Desktop\ASC_MinimumDistance\MATLAB_Artificial_Data\';
addpath 'C:\Users\mals6571\Desktop\ASC_MinimumDistance\Test_Images\DiffNoise\';
% XLabels = XLabels(1:512);
% LabelsFinal = LabelsFinal(1:512);

arirate = clustereval(XLabels,LabelsFinal, 'ari');
% arirate = clustereval(XLabels,LabelsFinal, 'ri');
disp('arirate = '); disp(arirate);

LabelsFinal1=bestMap(XLabels,LabelsFinal);
LabelsFinal1=LabelsFinal1';
hitrate = sum(XLabels(:) == LabelsFinal1(:)) / length(XLabels);
hitrate = hitrate * 100;
disp('hitrate = '); disp(hitrate);