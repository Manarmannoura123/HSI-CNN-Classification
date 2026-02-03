function classificationMap = generateMap(net, paddedData, rows, cols, bands, patchSize)
    halfPatch = floor(patchSize/2);
    classificationMap = zeros(rows, cols);
    
    fprintf('Generating full classification map. Please wait...\n');
    for i = 1:rows
        rowPatches = zeros(patchSize, patchSize, bands, cols);
        for j = 1:cols
            currR = i + halfPatch;
            currC = j + halfPatch;
            rowPatches(:,:,:,j) = paddedData(currR-halfPatch:currR+halfPatch, ...
                                             currC-halfPatch:currC+halfPatch, :);
        end
        rowPred = classify(net, rowPatches);
        classificationMap(i, :) = double(rowPred);
    end
end