function [X, Y, paddedData] = preprocessData(data, labels, patchSize, numSamplesPerClass)
    [rows, cols, bands] = size(data);
    halfPatch = floor(patchSize/2);
    
    % Min-Max Normalization
    data = (data - min(data(:))) / (max(data(:)) - min(data(:)));
    
    % Padding
    paddedData = padarray(data, [halfPatch, halfPatch, 0], 'symmetric');
    
    X = []; Y = [];
    fprintf('Extracting patches per class...\n');

    for classID = 1:16
        [r, c] = find(labels == classID);
        n = min(length(r), numSamplesPerClass);
        if n == 0, continue; end
        
        idx = randperm(length(r), n);
        classPatches = zeros(patchSize, patchSize, bands, n);
        
        for i = 1:n
            currR = r(idx(i)) + halfPatch;
            currC = c(idx(i)) + halfPatch;
            classPatches(:,:,:,i) = paddedData(currR-halfPatch:currR+halfPatch, ...
                                               currC-halfPatch:currC+halfPatch, :);
        end
        X = cat(4, X, classPatches);
        Y = [Y; repmat(classID, n, 1)];
    end
    Y = categorical(Y);
end