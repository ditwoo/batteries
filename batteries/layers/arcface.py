import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Implementation of ArcFace loss for metric learning.
    .. _ArcFace: Additive Angular Margin Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.07698v1

    Example:

        >>> layer = ArcFace(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ):
        """
        Args:
            embedding_size (int): size of each input sample.
            num_classes (int): size of each output sample.
            s (float, optional): norm of input feature,
                Default: ``64.0``.
            m (float, optional): margin.
                Default: ``0.5``.
            easy_margin (bool, optional): speed up hack.
                Default: ``False``.
        """
        super(ArcFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

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
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0.0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits
