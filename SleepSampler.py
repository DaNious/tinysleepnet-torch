import math

import numpy as np
from torch.utils.data import Sampler

class SleepSampler(Sampler):
    def __init__(self, 
                 dataset, 
                 groupSize=15, 
                 seqLength=20, 
                 isShuffle=True,
                 nightFirst=False, 
                 random_seed=None,
                 ):
        """ 
        Argument 'groupSize' determines how many nights' data is grouped to formulate a batch, 
        which will be used to generate the output index. 'groupSize' typically equals to batch size,
        while it sometimes has to be 1 (e.g., dataloader for testing and validation) so that will not
        be any dummy data (those padded with zeros).
        """
        self.dataset = dataset
        self.groupSize = groupSize
        self.seqLength = seqLength
        self.nightsNum = len(dataset)
        self.random_seed = random_seed
        self.isShuffle = isShuffle
        self.nightFirst = nightFirst
    
    def __iter__(self):
        # Whether or not shuffle night indices (determine if data is fed in night order)
        if not self.isShuffle:
            night_idx = np.arange(self.nightsNum)
        else:
            np.random.seed(self.random_seed)
            night_idx = np.random.permutation(np.arange(self.nightsNum))
        # Group subjects to find the max number of batches a single iteration of dataset can return
        n_group = int(math.ceil(len(night_idx) / self.groupSize))
        for i in range(n_group):
            # Find the max sleep epoch number in each group
            indexSel = night_idx[i*self.groupSize: (i+1)*self.groupSize].tolist()
            batchSel = [self.dataset.rawData_nights[idx] for idx in indexSel]
            maxEpochNum = max(item.shape[0] for item in batchSel)
            # Find the max number of sequences a single night data can return
            n_seq = int(math.ceil(maxEpochNum / self.seqLength))
            # Start yielding indices
            """
            There are two ways of generating a minibatch. If 'nightFirst' is False, the 1st minibatch 
            contains the beginning 20 epoches of every subject, the 2nd minibatch contains 21st to 40th
            epoches of every subject and so on. This is the way used in the original paper to build
            batches. Otherwise, if 'nightFirst' is True, the 1st minibatch contains only sleeping epoches
            of the first subject, which means the batch will iterate each subject's data first. 
            """
            if not self.nightFirst:
                for k in range(n_seq):
                    epoch_idx = k * self.seqLength
                    for j in range(self.groupSize):
                        if i * self.groupSize + j >= len(night_idx):
                            subject_idx = i * self.groupSize + j
                        else:
                            subject_idx = night_idx[i * self.groupSize + j]
                        yield subject_idx, epoch_idx
            else:
                for j in range(self.groupSize):
                    if i * self.groupSize + j >= len(night_idx):
                        subject_idx = i * self.groupSize + j
                    else:
                        subject_idx = night_idx[i * self.groupSize + j]
                    for k in range(n_seq):
                        epoch_idx = k * self.seqLength
                        yield subject_idx, epoch_idx