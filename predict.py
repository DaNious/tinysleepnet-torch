import os

import numpy as np
import sklearn.metrics as skmetrics

import torch
from torch.utils.data import DataLoader

from PSGdataset import PSGdataset
from SleepSampler import SleepSampler
from Model import TinySleepNet

BATCHSIZE = 15
DATADIR = 'data/sleepedf/eeg_fpz_cz'
DATASET = 'sleepedf'
MODELDIR = 'output'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict():
    # Load dataset and initialize dataloaders
    testSet = PSGdataset(DATASET, 'config/sleepedf_pred.txt', isAugment=False)
    print("----------Prediction----------")
    testSet.showInfo()
    testSampler = SleepSampler(testSet, groupSize=1, isShuffle=False)
    testLoader = DataLoader(testSet,
                            sampler=testSampler,
                            batch_size=BATCHSIZE,
                            drop_last=False)
    # Load model
    model = TinySleepNet().to(DEVICE)
    modelPath = os.path.join(MODELDIR, DATASET, 'best_model.pth')
    model.load_state_dict(torch.load(modelPath, weights_only=True))
    print(f"Load model from {modelPath}")

    # Start prediction
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch, (X, y, lstmReset) in enumerate(testLoader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            X = torch.reshape(X, (-1, 1, 3000))
            y = torch.reshape(y, (-1,))
            # Feed forward
            y_pred = model(X, bool(lstmReset[0]))
            # Save results
            labels_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
            labels_true = y.detach().cpu().numpy()
            preds.extend(labels_pred)
            trues.extend(labels_true)
    testAcc = skmetrics.accuracy_score(trues, preds) * 100
    testF1 = skmetrics.f1_score(trues, preds, average='macro') * 100
    testCM = skmetrics.confusion_matrix(trues, preds, labels=[0, 1, 2, 3, 4])
    print(f"Pred-acc: {testAcc:.1f} Pred-f1: {testF1:.1f} ")
    print(testCM)

if __name__ == "__main__":
    predict()