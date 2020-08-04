import pytest
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Subset, DataLoader

# local files
from batteries import engine, seed_all


IS_CUDA_PRESENT = torch.cuda.is_available()
if IS_CUDA_PRESENT:
    IS_MULTIPLE_CUDA_DEVICES = torch.cuda.device_count() > 1
else:
    IS_MULTIPLE_CUDA_DEVICES = False


class Projector(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)


def test_train_fn_on_cpu():
    seed_all(2020)

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
        train_loss = engine.train_fn(model, train_loader, device, criterion, optimizer)
        valid_loss = engine.valid_fn(model, valid_loader, device, criterion)
        metrics.append(
            {"epoch": epoch_idx + 1, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        print(f"[epoch {epoch_idx + 1}] loss - {train_loss:.5f}")
        print(f"[epoch {epoch_idx + 1}] loss - {valid_loss:.5f}")

    expected_train_loss = [
        0.6784567177295685,
        0.5936110734939575,
        0.5174714684486389,
        0.4503746211528778,
        0.392167529463768,
    ]
    expected_valid_loss = [
        0.6303888112306595,
        0.5502869486808777,
        0.479158878326416,
        0.41705310344696045,
        0.36359763741493223,
    ]

    assert len(metrics) == n_epochs
    assert np.allclose([m["epoch"] for m in metrics], list(range(1, n_epochs + 1)))
    assert np.allclose([m["train_loss"] for m in metrics], expected_train_loss)
    assert np.allclose([m["valid_loss"] for m in metrics], expected_valid_loss)


@pytest.mark.skipif(not IS_CUDA_PRESENT, reason="missing cuda")
def test_train_fn_on_cuda_0():
    seed_all(2020)

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
        train_loss = engine.train_fn(model, train_loader, device, criterion, optimizer)
        valid_loss = engine.valid_fn(model, valid_loader, device, criterion)
        metrics.append(
            {"epoch": epoch_idx + 1, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        print(f"[epoch {epoch_idx + 1}] loss - {train_loss:.5f}")
        print(f"[epoch {epoch_idx + 1}] loss - {valid_loss:.5f}")

    print(metrics)

    expected_train_loss = [
        0.6784567177295685,
        0.5936110734939575,
        0.5174714684486389,
        0.4503746211528778,
        0.392167529463768,
    ]
    expected_valid_loss = [
        0.6303888112306595,
        0.5502869486808777,
        0.479158878326416,
        0.41705310344696045,
        0.36359763741493223,
    ]

    assert len(metrics) == n_epochs
    assert np.allclose([m["epoch"] for m in metrics], list(range(1, n_epochs + 1)))
    assert np.allclose([m["train_loss"] for m in metrics], expected_train_loss)
    assert np.allclose([m["valid_loss"] for m in metrics], expected_valid_loss)


@pytest.mark.skipif(not IS_MULTIPLE_CUDA_DEVICES, reason="not enough cuda devices")
def test_train_fn_dp_cuda():
    seed_all(2020)

    n_records, n_features = int(1e4), 10
    batch_size = 32
    dataset = TensorDataset(torch.rand(n_records, n_features), torch.rand(n_records))
    dataset = Subset(dataset, list(range(batch_size * 10)))
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    valid_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    device = torch.device("cuda:0")
    model = nn.DataParallel(Projector(n_features)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    metrics = []
    n_epochs = 5
    for epoch_idx in range(n_epochs):
        train_loss = engine.train_fn(model, train_loader, device, criterion, optimizer)
        valid_loss = engine.valid_fn(model, valid_loader, device, criterion)
        metrics.append(
            {"epoch": epoch_idx + 1, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        print(f"[epoch {epoch_idx + 1}] loss - {train_loss:.5f}")
        print(f"[epoch {epoch_idx + 1}] loss - {valid_loss:.5f}")

    print(metrics)

    expected_train_loss = [
        0.6784567177295685,
        0.5936110734939575,
        0.5174714684486389,
        0.4503746211528778,
        0.392167529463768,
    ]
    expected_valid_loss = [
        0.6303888112306595,
        0.5502869486808777,
        0.479158878326416,
        0.41705310344696045,
        0.36359763741493223,
    ]

    assert len(metrics) == n_epochs
    assert np.allclose([m["epoch"] for m in metrics], list(range(1, n_epochs + 1)))
    assert np.allclose([m["train_loss"] for m in metrics], expected_train_loss)
    assert np.allclose([m["valid_loss"] for m in metrics], expected_valid_loss)
