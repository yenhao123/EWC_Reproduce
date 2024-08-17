import random
import torch
from torchvision import datasets

class PermutedMNIST(datasets.MNIST):

    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train=train, download=True)
        
        if permute_idx is None:
            permute_idx = torch.randperm(28 * 28)
        assert len(permute_idx) == 28 * 28
        
        # 使用 data 和 targets 屬性代替 train_data 和 test_data
        if self.train:
            self.data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                     for img in self.data])
        else:
            self.data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                     for img in self.data])

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [self.data[idx] for idx in sample_idx]
