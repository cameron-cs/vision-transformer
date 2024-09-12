import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class CIFAR10Dataset:
    """
    Wrapper class for CIFAR-10 dataset preparation, with transformations and optional data sampling.

    Args:
        path (str): The path to CIFAR10 data
        batch_size (int): The batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        train_sample_size (int): Optional, number of samples to use from the training set.
        test_sample_size (int): Optional, number of samples to use from the test set.
    """

    def __init__(self, path='./data', batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # transformations
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_train_loader(self):
        """
        Prepare and return the training data loader with optional sampling.
        """
        trainset = torchvision.datasets.CIFAR10(root=self.path, train=True, download=True,
                                                transform=self.train_transform)

        if self.train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:self.train_sample_size]
            trainset = Subset(trainset, indices)

        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return trainloader

    def get_test_loader(self):
        """
        Prepare and return the test data loader with optional sampling.
        """
        testset = torchvision.datasets.CIFAR10(root=self.path, train=False, download=True, transform=self.test_transform)

        if self.test_sample_size is not None:
            indices = torch.randperm(len(testset))[:self.test_sample_size]
            testset = Subset(testset, indices)

        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return testloader

    def get_classes(self):
        """
        Return the class labels for CIFAR-10 dataset.
        """
        return self.classes