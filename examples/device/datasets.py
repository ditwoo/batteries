from torch.utils.data import DataLoader, Subset

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def get_transforms(dataset: str):
    """Get transforms depends from dataset.

    Args:
        dataset (str): dataset type (train or valid)

    Returns:
        dataset transforms
    """
    return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


def get_loaders(stage: str) -> tuple:
    """Loaders for a stage.

    Args:
        stage (str): stage name

    Returns:
        training and validation loader
    """
    trainset = MNIST(
        "./data", train=False, download=True, transform=get_transforms("train"),
    )
    testset = MNIST(
        "./data", train=False, download=True, transform=get_transforms("valid"),
    )
    train = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
    valid = DataLoader(testset, batch_size=64, shuffle=True, num_workers=1)
    return train, valid
