from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        with gzip.open(image_filename, 'rb') as imgf:
            imgf.read(16)
            X = np.frombuffer(imgf.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32)
            X -= np.min(X)
            X /= np.max(X)
            self.imgs = X
        with gzip.open(label_filename, 'rb') as labelf:
            labelf.read(8)
            y = np.frombuffer(labelf.read(), dtype=np.uint8)
            self.labels = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if len(self.imgs[index].shape) >= 2:
            return np.array([self.apply_transforms(img.reshape(28, 28, 1)) for img in self.imgs[index]]), self.labels[index]
        else:
            return self.apply_transforms(self.imgs[index].reshape(28, 28, 1)), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.imgs)
        ### END YOUR SOLUTION