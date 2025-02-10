import os
import torch
from torch.utils.data import Dataset

import numpy as np

import collections

class PSGdataset(Dataset):
    def __init__(self, dataSetName, 
                 configFile, 
                 sigChannel='eeg_fpz_cz',
                 seqLength=20, 
                 isAugment=True,
                 augPercent = 0.1,
                 augMaxJump = 5,
                 random_seed=42,
                 ):
        self.SLEEPSTAGE = ['W', 'N1', 'N2', 'N3', 'R']
        self.seqLength = seqLength
        self.random_seed = random_seed
        # Read files listed in the .txt file
        with open(configFile, 'r') as f:
            fileNames = f.read().strip().split('\n')
        fileNames.sort()
        for fileName in fileNames:
            if fileName.split('.')[-1] != 'npz':
                fileNames.remove(fileName)
        self.rawData_nights = []
        self.label_nights = []
        for npzFile in fileNames:
            with np.load(os.path.join('data', dataSetName, sigChannel, npzFile)) as f:
                assert f['x'].shape[0] == f['y'].shape[0], f"Data and label in file {npzFile} do not match!"
                self.rawData_nights.append(np.squeeze(f['x']))
                self.label_nights.append(np.squeeze(f['y']))
        assert len(self.rawData_nights) == len(self.label_nights), f"Dataset has nonequivalent night-label pairs!"
        # Augment the data by random shifting and start jumping
        if isAugment: self.dataAugment(shiftPercent=augPercent, maxJump=augMaxJump)
    
    def __len__(self):
        return len(self.rawData_nights)

    def __getitem__(self, index):
        """
        Return zeros as raw data and labels if the index is out of range. (Not sure the point of 
        having those raw data and labels as zeros, which represents W sleeping stage, 
        perhaps it's related to dataset balancing and training strategy in the original paper.)
        """
        subject_idx = index[0]
        epoch_idx = index[1]
        # Flag to reset LSTM states
        lstmReset = True if epoch_idx == 0 else False
        # Numpy array shapes as (sequence length, 30 * sampling rate) and (sequence length, )
        rawSeq = np.zeros((self.seqLength, self.rawData_nights[0].shape[1]), dtype=np.float32)
        labelSeq = np.zeros(self.seqLength, dtype=np.int32)
        # Return zero array if request index is out of range
        if subject_idx < len(self.rawData_nights):
            epoch_idx_start = epoch_idx
            epoch_idx_end = epoch_idx + self.seqLength
            # Assign values to the returned array, leave out-range indexed values as zeros
            if epoch_idx_end < self.rawData_nights[subject_idx].shape[0]:
                rawSeq = self.rawData_nights[subject_idx][epoch_idx_start: epoch_idx_end]
                labelSeq = self.label_nights[subject_idx][epoch_idx_start: epoch_idx_end]
            else:
                # Put the remainder (epoch num % sequence length) into the returned array, leave rest values as zeros
                rawRemainder = self.rawData_nights[subject_idx][epoch_idx_start:]
                labelRemainder = self.label_nights[subject_idx][epoch_idx_start:]
                rawSeq[ :len(rawRemainder), :] = rawRemainder
                labelSeq[ :len(labelRemainder)] = labelRemainder
        return torch.Tensor(rawSeq[:, np.newaxis, :]), torch.Tensor(labelSeq).long(), lstmReset
    
    def dataAugment(self, shiftPercent=0.1, maxJump=5):
        np.random.seed(self.random_seed)
        for i in range(len(self.rawData_nights)):
            # 1. Shift each night data by defined percentage
            offset = np.random.uniform(-shiftPercent, shiftPercent) * self.rawData_nights[i].shape[1]
            # Discard the first or last epoch (depends on the shifting direction)
            if offset < 0:
                self.rawData_nights[i] = np.roll(self.rawData_nights[i], int(offset))[:-1]
                self.label_nights[i] = self.label_nights[i][:-1]
            if offset > 0:
                self.rawData_nights[i] = np.roll(self.rawData_nights[i], int(offset))[1:]
                self.label_nights[i] = self.label_nights[i][1:]
            assert len(self.rawData_nights) == len(self.label_nights), f"Dataset has nonequivalent night-label pairs after shifting augmentation!"
            # 2. Discard certain epochs at the beginning
            epochJump = np.random.randint(maxJump)
            self.rawData_nights[i] = self.rawData_nights[i][epochJump:]
            self.label_nights[i] = self.label_nights[i][epochJump:]
            assert len(self.rawData_nights) == len(self.label_nights), f"Dataset has nonequivalent night-label pairs after jumping augmentation!"
    
    def showInfo(self):
        stageCnt = [0] * 5
        for labels in self.label_nights:
            for i in range(5):
                stageCnt[i] = stageCnt[i] + np.count_nonzero(labels == i)
        print(f"{len(self.rawData_nights)} night(s) data included")
        for i in range(5):
            print(f"Stage: {self.SLEEPSTAGE[i]:<2} - {stageCnt[i]}")


if __name__ == '__main__':
    dataset = PSGdataset('sleepedf', 'config/sleepedf_test.txt')
    for data, label in zip(dataset.rawData_nights, dataset.label_nights):
        print(data.shape, label.shape)
    print(dataset.getInfo())