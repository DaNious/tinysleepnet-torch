# TinySleepNet (PyTorch implementation)

This is the PyTorch implementation of the work [TinySleepNet](https://github.com/akaraspt/tinysleepnet) (written by TensorFlow 1.x). Compared with other PyTorch implementations on Github, this repo fits to the PyTorch 2.x structure and coding manner better with following points:
1. When feeding data, it uses native torch dataset, dataloader classes with a self-defined index sampler to ensure the feeding order is still same as the original work.
2. For training, corresponding PyTorch classes(loss, gradient,.etc.) are used instead of doing a lot of manual functional coding.

## How to run
1. Create a virtual environment
```python
conda create -n tinysleepnet python=3.11
conda activate tinysleepnet
pip install -r requirements.txt
```
2. Get the dataset
    1. Download the original dataset from [Sleep-EDF Database Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/)
    2. Prepare the dataset to get *.npz format data files using https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py
    3. In ***train.py***, change DATADIR to your data directory.
    4. Use **datasetSplit** function to generate *.txt files under /config that configurate the training, validation and testing data.
    5. Change training settings as needed and run:
    ```python
    python train.py
    ```
