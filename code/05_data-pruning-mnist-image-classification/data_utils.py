import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

import data_pruning


def _create_subset_dataloader(train_dataset, indices, batch_size=64, shuffle=True):
    pruned_dataset = Subset(train_dataset, indices)
    dataloader = DataLoader(pruned_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def _get_transforms(is_train=True):
    # Normalization from LeNet-5 paper
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
            ]
        )


def _load_raw_mnist(download_path="./data"):
    train_dataset = torchvision.datasets.MNIST(
        root=download_path,
        train=True,
        download=True,
        transform=_get_transforms(is_train=True),
    )
    test_dataset = torchvision.datasets.MNIST(
        root=download_path,
        train=False,
        download=True,
        transform=_get_transforms(is_train=False),
    )

    return train_dataset, test_dataset


def get_data_loaders(pruning_method="none", batch_size=64):
    train_dataset, test_dataset = _load_raw_mnist()
    indices_to_keep = data_pruning.prune_indices(train_dataset, method=pruning_method)
    train_loader = _create_subset_dataloader(
        train_dataset, indices_to_keep, batch_size=batch_size
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
