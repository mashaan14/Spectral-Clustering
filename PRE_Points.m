load('Zelnik-Manor-Data.mat');
X = XX{4};
XLabels = csvread('Sparse622.csv'); % TwoSpiral1000 % 3rings299 % dbmoon1000 % smile266 % lines512 % clusters300 % Sparse622 % Ring238 % Sparse303
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('data_Aggregation.mat'); % C=7
% load('data_Bridge.mat'); % C=2
% load('data_Compound.mat'); % C=6
% load('data_Flame.mat'); % C=2
% load('data_Jain.mat'); % C=2
% load('data_Spiral.mat'); % C=3
% load('data_TwoDiamonds.mat'); % C=2
% X = D;
% XLabels = L;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('Zelnik-LinesNoiseAdd.mat');
% X = S.XNoise0p0;
% XLabels = S.XNoise0p0L;

% figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3); pbaspect([1 1 1]); daspect([1 1 1]); axis off;