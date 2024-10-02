import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset
import gzip

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        if train:
            data_batches = [os.path.join(base_folder, f'data_batch_{i}') for i in range(1, 6)]
        else:
            data_batches = [os.path.join(base_folder, 'test_batch')]
        self.X = []
        self.y = []
        for batch in data_batches:
            with open(batch, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
            self.X.append(data[b'data'])
            self.y.append(data[b'labels'])
        self.X = np.concatenate(self.X)
        self.y = np.concatenate(self.y)
        self.X = self.X / 255.0
        self.X = self.X.reshape(-1, 3, 32, 32)
        self.p = p
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X = self.X[index]
        y = self.y[index]
        if self.transforms:
            for transform in self.transforms:
                X = transform(X)
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
