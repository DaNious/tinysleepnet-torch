# TinySleepNet (PyTorch implementation)

This is the PyTorch implementation of the work [TinySleepNet](https://github.com/akaraspt/tinysleepnet) (written by TensorFlow 1.x). Compared with other PyTorch implementations on Github, this repo fits to the PyTorch 2.x structure and coding manner better with following points:
1. When feeding data, it uses native torch dataset, dataloader classes with a self-defined index sampler to ensure the feeding order is still same as the original work.
2. For training, corresponding PyTorch classes(loss, gradient,.etc.) are used instead of doing a lot of manual functional coding.