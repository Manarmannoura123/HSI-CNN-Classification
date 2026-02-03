function layers = getCNNArchitecture(patchSize, numBands, numClasses)
    layers = [
        imageInputLayer([patchSize patchSize numBands], 'Name', 'input')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_1')
        batchNormalizationLayer('Name', 'bn_1')
        reluLayer('Name', 'relu_1')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2')
        batchNormalizationLayer('Name', 'bn_2')
        reluLayer('Name', 'relu_2')
        
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool_max')
        
        fullyConnectedLayer(128, 'Name', 'fc_1')
        reluLayer('Name', 'relu_3')
        dropoutLayer(0.5, 'Name', 'dropout')
        
        fullyConnectedLayer(numClasses, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'class_output')
    ];
end