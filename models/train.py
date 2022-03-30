import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from dataset import MultiDirectoryDataSequence
from model import DAVE2


def train(
    model: nn.Module,
    dataloader: data.DataLoader,
    optimizer: optim.Optimizer,
    device=torch.device("cpu"),
    log_freq=100,
):
    running_loss = 0.0
    for i, (x, y) in enumerate(dataloader, 1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_ = model(x)
        loss = F.mse_loss(y_, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_freq == 0:
            print(f"  iteration={i}: loss={loss.item()}")
    return running_loss


def main():
    num_epochs = 100
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiDirectoryDataSequence(
        "~/Data/BeamNG", transform=transforms.Compose([transforms.ToTensor()])
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    print(f"Dataset has {len(dataset)} images")

    model = DAVE2()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, num_epochs + 1):
        print(f"EPOCH={epoch}")
        start_t = time.time()
        total_loss = train(model, dataloader, optimizer, device=device)
        end_t = time.time()
        print(f"  average epoch loss: {total_loss / len(dataset)}")
        print(f"  epoch time: {end_t - start_t}")
        torch.save(model, f"test_model_{epoch}.pt")


if __name__ == "__main__":
    main()
