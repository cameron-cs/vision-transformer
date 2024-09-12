from vit.dataset.cifar10_dataset import CIFAR10Dataset


def prepare_cifar10_data(path='./data', batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    """
    Wrapper function to prepare CIFAR-10 data using CIFAR10Dataset class.

    Args:
        path (str): The path to CIFAR10 data
        batch_size (int): The batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        train_sample_size (int): Optional, number of samples to use from the training set.
        test_sample_size (int): Optional, number of samples to use from the test set.

    Returns:
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        classes (tuple): CIFAR-10 class names.
    """

    cifar10 = CIFAR10Dataset(path=path, batch_size=batch_size, num_workers=num_workers, train_sample_size=train_sample_size,
                             test_sample_size=test_sample_size)

    train_loader = cifar10.get_train_loader()
    test_loader = cifar10.get_test_loader()
    classes = cifar10.get_classes()

    return train_loader, test_loader, classes
