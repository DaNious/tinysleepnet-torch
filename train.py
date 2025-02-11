import os

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics

import torch
import torch.optim.adam
from torch.utils.data import DataLoader
from torch import nn

from PSGdataset import PSGdataset
from SleepSampler import SleepSampler
from Model import TinySleepNet
from utils import datasetSplit, computeLoss

BATCHSIZE = 15
DATADIR = 'data/sleepedf/eeg_fpz_cz'
MODELDIR = 'output'
DATASET = 'sleepedf'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSSWEIGHT = torch.Tensor([1., 1.5, 1., 1., 1.])
LR = 1e-4
WEIGHTDECAY = 1e-3
TRAIN_EPOCH = 200
GRADCLIP = 5.0

def train():
    # Load dataset and initialize dataloaders
    trainSet = PSGdataset(DATASET, 'config/sleepedf_train.txt', isAugment=True)
    print("----------Training----------")
    trainSet.showInfo()
    trainSampler = SleepSampler(trainSet, groupSize=BATCHSIZE)
    trainLoader = DataLoader(trainSet,
                            sampler=trainSampler,
                            batch_size=BATCHSIZE,
                            drop_last=False)
    
    validSet = PSGdataset(DATASET, 'config/sleepedf_valid.txt', isAugment=False)
    print("----------Validation----------")
    validSet.showInfo()
    validSampler = SleepSampler(validSet, groupSize=1, isShuffle=False)
    validLoader = DataLoader(validSet,
                            sampler=validSampler,
                            batch_size=BATCHSIZE,
                            drop_last=False)
    
    testSet = PSGdataset(DATASET, 'config/sleepedf_test.txt', isAugment=False)
    print("----------Testing----------")
    testSet.showInfo()
    testSampler = SleepSampler(testSet, groupSize=1, isShuffle=False)
    testLoader = DataLoader(testSet,
                            sampler=testSampler,
                            batch_size=BATCHSIZE,
                            drop_last=False)
    
    # Prepare for training
    model = TinySleepNet().to(DEVICE)
    print("----------Model Info----------")
    print(model)

    # Loss function (Use weight 1.5 for N1 stage), PyTorch performs the reduction to get the mean by default
    lossFunc = nn.CrossEntropyLoss(weight=LOSSWEIGHT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    bestAcc, bestF1 = -1, -1
    # Start training
    print("----------Start training----------")
    for i in range(TRAIN_EPOCH + 1):
        model.train()
        losses, preds, trues = [], [], []
        for batch, (X, y, lstmReset) in enumerate(trainLoader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            X = torch.reshape(X, (-1, 1, 3000))
            y = torch.reshape(y, (-1,))
            # Feed forward
            y_pred = model(X, bool(lstmReset[0]))
            # Compute loss (cross-entropy + regularization loss)
            loss = computeLoss(y_pred, y, model, lossFunc, isTraining=True, weightDecay=WEIGHTDECAY)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), clip_value=GRADCLIP)
            optimizer.step()
            # Save results
            losses.append(loss.detach().cpu())
            labels_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
            labels_true = y.detach().cpu().numpy()
            preds.extend(labels_pred)
            trues.extend(labels_true)
        trainLoss = np.array(losses).mean()
        trainAcc = skmetrics.accuracy_score(trues, preds) * 100
        trainF1 = skmetrics.f1_score(trues, preds, average='macro') * 100
        # Validation
        model.eval()
        losses, preds, trues = [], [], []
        with torch.no_grad():
            for batch, (X, y, lstmReset) in enumerate(validLoader):
                X, y = X.to(DEVICE), y.to(DEVICE)
                X = torch.reshape(X, (-1, 1, 3000))
                y = torch.reshape(y, (-1,))
                # Feed forward
                y_pred = model(X, bool(lstmReset[0]))
                # Compute loss
                loss = computeLoss(y_pred, y, model, lossFunc, isTraining=False)
                # Save results
                losses.append(loss.detach().cpu())
                labels_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
                labels_true = y.detach().cpu().numpy()
                preds.extend(labels_pred)
                trues.extend(labels_true)
        validLoss = np.array(losses).mean()
        validAcc = skmetrics.accuracy_score(trues, preds) * 100
        validF1 = skmetrics.f1_score(trues, preds, average='macro') * 100
        # Testing
        model.eval()
        losses, preds, trues = [], [], []
        with torch.no_grad():
            for batch, (X, y, lstmReset) in enumerate(testLoader):
                X, y = X.to(DEVICE), y.to(DEVICE)
                X = torch.reshape(X, (-1, 1, 3000))
                y = torch.reshape(y, (-1,))
                # Feed forward
                y_pred = model(X, bool(lstmReset[0]))
                # Compute loss
                loss = computeLoss(y_pred, y, model, lossFunc, isTraining=False)
                # Save results
                losses.append(loss.detach().cpu())
                labels_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
                labels_true = y.detach().cpu().numpy()
                preds.extend(labels_pred)
                trues.extend(labels_true)
        testLoss = np.array(losses).mean()
        testAcc = skmetrics.accuracy_score(trues, preds) * 100
        testF1 = skmetrics.f1_score(trues, preds, average='macro') * 100
        testCM = skmetrics.confusion_matrix(trues, preds, labels=[0, 1, 2, 3, 4])
        print(f"Epoch: {i:<4} |"
                f" TR-loss: {trainLoss:.4f} TR-acc: {trainAcc:.1f} TR-f1: {trainF1:.1f} | "
                f" VA-loss: {validLoss:.4f} VA-acc: {validAcc:.1f} VA-f1: {validF1:.1f} | "
                f" TE-loss: {testLoss:.4f} TE-acc: {testAcc:.1f} TE-f1: {testF1:.1f} ")
        print(testCM)
        # Save the model with best performance on validation set
        if bestAcc < validAcc and bestF1 < validF1:
            bestAcc = validAcc
            bestF1 = validF1
            saveDir = os.path.join(MODELDIR, DATASET)
            if not os.path.exists(saveDir): os.makedirs(saveDir)
            print(f"Save best model at epoch {i}.")
            torch.save(model.state_dict(), os.path.join(saveDir, 'best_model.pth'))

if __name__ == "__main__":
    # Generate .txt files to split the dataset into training, validation and testing
    datasetSplit(os.listdir(DATADIR), 
                 DATASET, 
                 byID=True, 
                 foldNum=10, 
                 foldIdx=0, 
                 validPercent=0.1, 
                #  random_seed=42,
                 )
    train()