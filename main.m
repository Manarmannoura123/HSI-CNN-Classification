%% MAIN SCRIPT: Salinas HSI Classification
clear; clc; close all;

% 1. Settings
patchSize = 11;
numSamplesPerClass = 200;

% 2. Load Data
load('Salinas.mat'); load('Salinas_gt.mat');
data = double(salinas); labels = double(salinas_gt);
[rows, cols, bands] = size(data);

% 3. Preprocess & Extract Patches (Calls preprocessData.m)
[X, Y, paddedData] = preprocessData(data, labels, patchSize, numSamplesPerClass);

% Shuffle and Split
idx = randperm(size(X,4));
X = X(:,:,:,idx); Y = Y(idx);
splitIdx = floor(0.8 * size(X,4));
XTrain = X(:,:,:,1:splitIdx); YTrain = Y(1:splitIdx);
XVal = X(:,:,:,splitIdx+1:end); YVal = Y(splitIdx+1:end);

% 4. Build Architecture (Calls getCNNArchitecture.m)
layers = getCNNArchitecture(patchSize, bands, 16);

% 5. Train Model
options = trainingOptions('adam', 'MaxEpochs', 25, 'MiniBatchSize', 64, ...
    'ValidationData', {XVal, YVal}, 'Plots', 'training-progress', 'Verbose', false);
net = trainNetwork(XTrain, YTrain, layers, options);

% 6. Inference (Calls generateMap.m)
classificationMap = generateMap(net, paddedData, rows, cols, bands, patchSize);

% 7. Visualize Results
figure;
subplot(1,2,1); imagesc(labels); title('Ground Truth'); axis image;
subplot(1,2,2); imagesc(classificationMap); title('CNN Prediction'); axis image;
colormap(jet(17)); colorbar;