% Uncomment one dataset to use

load('Mashaan-CircleSinCos.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('Zelnik-Manor-Data.mat');
% X = XX{4};
% XLabels = csvread('Sparse622.csv'); % Sparse622 % Ring238 % Sparse303
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(X(:,1),X(:,2),'o','Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'MarkerSize',3); pbaspect([1 1 1]); daspect([1 1 1]); axis off;