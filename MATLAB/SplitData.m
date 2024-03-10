function SplitData(dataPath, trainPath, valPath)
% SPLITDATA Split the images in separate folders randomly with a
%           ratio of 80%
%
% Toan Ly
% 2/18/2024
%

files = dir(fullfile(dataPath, '*.mat'));

rng('shuffle');
indices = randperm(length(files));

numTrain = round(length(files) * 0.8);

trainFiles = files(indices(1:numTrain));
valFiles = files(indices(numTrain+1:end));

for i = 1:length(trainFiles)
    filename = trainFiles(i).name;
    movefile(fullfile(dataPath, filename), fullfile(trainPath, filename));
end

for i = 1:length(valFiles)
    filename = valFiles(i).name;
    movefile(fullfile(dataPath, filename), fullfile(valPath, filename));
end


end