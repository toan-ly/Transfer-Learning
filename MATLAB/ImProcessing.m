mean_subtract_val = [103.939, 116.779, 123.68];

%% Store Doraemon
dataPath = 'Images\Doraemon';
outPath =  'Data';
StoreImages(dataPath, outPath, mean_subtract_val, 1);

%% Store Oggy
dataPath = 'Images\Oggy';
StoreImages(dataPath, outPath, mean_subtract_val, 0);

%% Split into Train and Val
trainPath = 'Data\Train';
valPath = 'Data\Val';
dataPath = outPath;
SplitData(dataPath, trainPath, valPath);





