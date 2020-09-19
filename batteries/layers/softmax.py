import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMax(nn.Module):
    """Implementation of SoftMax head for metric learning.

    Example:

        >>> layer = SoftMax()
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(self, embedding_size: int, num_classes: int):
        """
        Args:
            embedding_size (int): size of each input sample.
            num_classes (int): size of each output sample.
        """
        super(SoftMax, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        self.bias = nn.Parameter(torch.FloatTensor(num_classes))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input features,
                expected shapes BxF.

        Returns:
            torch.Tensor with loss value.
        """
        return F.linear(input, self.weight, self.bias)
