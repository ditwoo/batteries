import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """Implementation of Arc Margin Product.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = ArcMarginProduct(5, 10)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: size of each input sample.
            out_features: size of each output sample.
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "ArcMarginProduct("
            f"in_features={self.in_features},"
            f"out_features={self.out_features}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine


class SoftMax(nn.Module):
    """Implementation of
    `Significance of Softmax-based Features in Comparison to
    Distance Metric Learning-based Features`_.

    .. _Significance of Softmax-based Features in Comparison to \
        Distance Metric Learning-based Features:
        https://arxiv.org/abs/1712.10151

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = SoftMax(5, 10)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(self, in_features: int, num_classes: int):
        """
        Args:
            in_features: size of each input sample.
            out_features: size of each output sample.
        """
        super(SoftMax, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        self.bias = nn.Parameter(torch.FloatTensor(num_classes))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "SoftMax("
            f"in_features={self.in_features},"
            f"out_features={self.out_features}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        return F.linear(input, self.weight, self.bias)
