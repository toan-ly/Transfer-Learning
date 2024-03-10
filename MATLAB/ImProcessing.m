mean_subtract_val = [103.939, 116.779, 123.68];

%% Store Doraemon
dataPath = 't:\datatransfer\Interns\Toan\Lessons\9-TransferLearning\Images\Doraemon';
outPath =  't:\datatransfer\Interns\Toan\Lessons\9-TransferLearning\Data';
StoreImages(dataPath, outPath, mean_subtract_val, 1);

%% Store Oggy
dataPath = 't:\datatransfer\Interns\Toan\Lessons\9-TransferLearning\Images\Oggy';
StoreImages(dataPath, outPath, mean_subtract_val, 0);

%% Split into Train and Val
trainPath = 't:\datatransfer\Interns\Toan\Lessons\9-TransferLearning\Data\Train';
valPath = 't:\datatransfer\Interns\Toan\Lessons\9-TransferLearning\Data\Val';
dataPath = outPath;
SplitData(dataPath, trainPath, valPath);





