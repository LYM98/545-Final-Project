% Read image 
img = imread('demo1.jpg');

% Set parameters of GIST
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing GIST
[gist, param] = LMgist(img, '', param);