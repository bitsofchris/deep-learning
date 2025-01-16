# data_loader.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_transforms(is_train=True):
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


def load_data(batch_size):
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=get_transforms(is_train=True),
        download=True,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=get_transforms(is_train=False),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader
