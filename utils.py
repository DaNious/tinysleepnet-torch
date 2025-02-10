import numpy as np

import torch

def datasetSplit(fileNames,
                 dataset, 
                 byID=False, 
                 foldNum=10, 
                 foldIdx=0, 
                 validPercent=0.1, 
                 random_seed=42
                 ):
    fileNames.sort()
    # Whether or not split the dataset by subject ID
    if byID:
        # Find subject IDs
        subjectID = []
        for item in fileNames:
            subjectID.append(int(item.split('.')[0][3:5]))
        subjectID_unique = np.asarray(list(set(subjectID)))
        # Split train, valid and test ID for selected fold
        foldIDs = np.array_split(subjectID_unique, foldNum)
        testIDs = foldIDs[foldIdx]
        trainIDs = np.setdiff1d(subjectID_unique, testIDs)
        np.random.seed(random_seed)
        validIDs = np.random.choice(trainIDs, size=round(len(trainIDs) * validPercent), replace=False)
        trainIDs = np.setdiff1d(trainIDs, validIDs)
        # Find nights' data using IDs
        trainNights = []
        for id in trainIDs:
            indices = np.where(subjectID == id)[0]
            for idx in indices:
                trainNights.append(fileNames[idx])
        validNights = []
        for id in validIDs:
            indices = np.where(subjectID == id)[0]
            for idx in indices:
                validNights.append(fileNames[idx])
        testNights = []
        for id in testIDs:
            indices = np.where(subjectID == id)[0]
            for idx in indices:
                testNights.append(fileNames[idx])
        trainNights = np.array(trainNights)
        validNights = np.array(validNights)
        testNights = np.array(testNights)
    else:
        # Split train, valid and test nights' data for selected fold
        foldNights = np.array_split(fileNames, foldNum)
        testNights = foldNights[foldIdx]
        trainNights = np.setdiff1d(fileNames, testNights)
        np.random.seed(random_seed)
        validNights = np.random.choice(trainNights, size=round(len(trainNights) * validPercent), replace=False)
        trainNights = np.setdiff1d(trainNights, validNights)
    # Save to .txt as config files
    np.savetxt('config/' + dataset + '_train.txt', trainNights, fmt='%s')
    np.savetxt('config/' + dataset + '_valid.txt', validNights, fmt='%s')
    np.savetxt('config/' + dataset + '_test.txt', testNights, fmt='%s')

def computeLoss(y_pred, y, model, lossFunc, isTraining=True, weightDecay=1e-3):
    # cross-entropy + regularization loss
    convWeights = [param for name, param in model.named_parameters() if 'conv' in name]
    lossReg = 0
    for p in convWeights:
        lossReg = lossReg + torch.sum(p ** 2) * weightDecay
    lossCE = lossFunc(y_pred, y)
    lossTotal = lossCE + lossReg
    if isTraining:
        return lossTotal
    else:
        return lossCE