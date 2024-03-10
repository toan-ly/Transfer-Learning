function StoreImages(dataPath, outPath, mean_subtract_val, label)
% STOREIMAGE Get images from input file, preprocessing and store 
%            the image and its label in output file
%
% Toan Ly
% 2/16/2024
%

files = dir(fullfile(dataPath, 'm*.*'));
for i = 1:length(files)
    filename = fullfile(dataPath, files(i).name);
    im = single(imread(filename));
    for j = 1:3
        im(:, :, j) = im(:, :, j) - mean_subtract_val(j);
    end
    im = imresize(im, [224, 224]);
    save(fullfile(outPath, [files(i).name(1:4) '.mat']), 'im', 'label');
end



end