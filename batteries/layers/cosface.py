import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """Implementation of CosFace loss for metric learning.
    .. _CosFace: Large Margin Cosine Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.09414

    Example:

        >>> layer = CosFaceLoss(5, 10, s=1.31, m=0.1)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(
        self, embedding_size: int, num_classes: int, s: float = 64.0, m: float = 0.35,
    ):
        """
        Args:
            embedding_size (int): size of each input sample.
            num_classes (int): size of each output sample.
            s (float, optional): norm of input feature,
                Default: ``64.0``.
            m (float, optional): margin.
                Default: ``0.35``.
        """
        super(CosFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.projection = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.projection)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input features,
                expected shapes BxF.
            target (torch.Tensor): target classes,
                expected shapes B.

        Returns:
            torch.Tensor with loss value.
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.projection))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits
