import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    r"""Implementation of
    `Root Mean Square Layer Normalization`

    .. _Root Mean Square Layer Normalization:
        https://arxiv.org/abs/1910.07467

    Example:
        >>> layer = RMSNorm(10)
        >>> x = torch.randn(3, 5, 10)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([3, 5, 10])

    """

    def __init__(self, in_features: int, partial: float = None, bias: bool = False, eps: float = 1e-8):
        """
        Args:
            in_features: dimension of input features.
                For example, if input tensor have shape [B, ... , F] then ``in_feautures`` should be equal to F.
            partial: option to use partial RMSNorm, to enable this features pass a value in a range [0.0, 1.0].
                If ``None`` then partial RMSNorm will be disabled.
                Default is ``None``.
            bias: option to use bias centering.
                Default is ``False``.
            eps: epsion value, used in division.
                Default is ``1e-8``.

        """
        super().__init__()

        if partial is None:
            partial = 0.0
        if not (0.0 <= partial <= 1.0):
            raise ValueError("Partial should be in a range [0.0, 1.0]!")
        self.partial = partial

        self.in_features = in_features
        self.scale = nn.Parameter(torch.ones(in_features))
        self.bias = bias
        self.eps = eps

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(in_features))

    def __repr__(self) -> str:
        """Object representation."""
        # fmt: off
        rep = (
            "RMSNorm("
            f"in_features={self.in_features},"
            f"parial={self.partial},"
            f"bias={self.bias},"
            f"eps={self.eps}"
            ")"
        )
        # fmt: on
        return rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features, expected shapes - [B, ... , in_features].

        Returns:
            torch.Tensor with normalized features with shape [B, ... , in_features].

        """
        if self.partial > 0.0:
            x_size = int(self.partial * self.in_features)
            x_part, _ = torch.split(x, [x_size, self.in_features - x_size], dim=-1)

            x_norm = torch.norm(x_part, p=2, dim=-1, keepdim=True)
        else:
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            x_size = self.in_features

        rms_x = x_norm * x_size ** (-0.5)
        res = x / (rms_x + self.eps)

        res = self.scale * res
        if self.bias:
            res = res + self.offset

        return res
