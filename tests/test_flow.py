import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Subset, DataLoader

from batteries import tv
from batteries import seeds

IS_CUDA_PRESENT = torch.cuda.is_available()


class Projector(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)


def test_train_fn_on_cpu():
    seeds.seed_all(2020)

    n_records, n_features = int(1e4), 10
    batch_size = 32
    dataset = TensorDataset(torch.rand(n_records, n_features), torch.rand(n_records))
    dataset = Subset(dataset, list(range(batch_size * 10)))
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    valid_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    device = torch.device("cpu")
    model = Projector(n_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    metrics = []
    n_epochs = 5
    for epoch_idx in range(n_epochs):
        train_loss = tv.train_fn(model, train_loader, device, criterion, optimizer)
        valid_loss = tv.valid_fn(model, valid_loader, device, criterion)
        metrics.append(
            {"epoch": epoch_idx + 1, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        print(f"[epoch {epoch_idx + 1}] loss - {train_loss:.5f}")
        print(f"[epoch {epoch_idx + 1}] loss - {valid_loss:.5f}")

    expected = [
        {
            "epoch": 1,
            "train_loss": 0.6784567177295685,
            "valid_loss": 0.6303887814283371,
        },
        {
            "epoch": 2,
            "train_loss": 0.5936110615730286,
            "valid_loss": 0.5502869486808777,
        },
        {
            "epoch": 3,
            "train_loss": 0.5174714475870132,
            "valid_loss": 0.47915886640548705,
        },
        {
            "epoch": 4,
            "train_loss": 0.4503746122121811,
            "valid_loss": 0.4170530676841736,
        },
        {
            "epoch": 5,
            "train_loss": 0.3921675056219101,
            "valid_loss": 0.36359761357307435,
        },
    ]

    assert len(metrics) == n_epochs
    assert metrics == expected


@pytest.mark.skipif(not IS_CUDA_PRESENT, reason="missing cuda")
def test_train_fn_on_cuda_0():
    seeds.seed_all(2020)

    n_records, n_features = int(1e4), 10
    batch_size = 32
    dataset = TensorDataset(torch.rand(n_records, n_features), torch.rand(n_records))
    dataset = Subset(dataset, list(range(batch_size * 10)))
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    valid_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    device = torch.device("cuda:0")
    model = Projector(n_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    metrics = []
    n_epochs = 5
    for epoch_idx in range(n_epochs):
        train_loss = tv.train_fn(model, train_loader, device, criterion, optimizer)
        valid_loss = tv.valid_fn(model, valid_loader, device, criterion)
        metrics.append(
            {"epoch": epoch_idx + 1, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        print(f"[epoch {epoch_idx + 1}] loss - {train_loss:.5f}")
        print(f"[epoch {epoch_idx + 1}] loss - {valid_loss:.5f}")

    print(metrics)

    expected = [
        {
            "epoch": 1,
            "train_loss": 0.6784567177295685,
            "valid_loss": 0.6303888112306595,
        },
        {
            "epoch": 2,
            "train_loss": 0.5936110734939575,
            "valid_loss": 0.5502869486808777,
        },
        {
            "epoch": 3,
            "train_loss": 0.5174714595079422,
            "valid_loss": 0.4791588693857193,
        },
        {
            "epoch": 4,
            "train_loss": 0.45037461519241334,
            "valid_loss": 0.4170530825853348,
        },
        {
            "epoch": 5,
            "train_loss": 0.3921675205230713,
            "valid_loss": 0.3635976195335388,
        },
    ]

    assert len(metrics) == n_epochs
    assert metrics == expected
