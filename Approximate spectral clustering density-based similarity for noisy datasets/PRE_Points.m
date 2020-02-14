addpath 'C:\Users\mals6571\Desktop\ASC_MinimumDistance\MATLAB_Artificial_Data';
%% =======================================================
% x = linspace(0,0.01,100); y = 1*ones(1,length(x)); f1 = [x;y];
% R = normrnd(0,0.050,1,length(x));
% f1 = f1+R;
% x = linspace(0,0.01,100); y = 2*ones(1,length(x)); f2 = [x;y];
% f2 = f2+R;
% x = linspace(0,0.01,100); y = 3*ones(1,length(x)); f3 = [x;y];
% f3 = f3+R;
% x = linspace(0,0.01,100); y = 4*ones(1,length(x)); f4 = [x;y];
% f4 = f4+R;
% X = [f1'; f2'; f3'; f4'];
% load('lines400s=0p150.mat');
% XLabels = csvread('lines400.csv');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NoisyData = Data + sigma*randn(size(Data));
% f0p01 = X + 0.01*randn(size(X));
% load('Zelnik-SmileNoise.mat'); % Zelnik-SmileNoise % Zelnik-RingsNoise % Zelnik-LinesNoise
% X = S.f0p05;
% XLabels = csvread('lines512.csv');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('Zelnik-Manor-Data.mat');
X = XX{1};
XLabels = csvread('3rings299.csv'); % 1-3rings299 % 3-smile266 % 5-lines512 % 6-circle238
% for XX{5} only
% X = [X; [X(:,1)+1 X(:,2)]];
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