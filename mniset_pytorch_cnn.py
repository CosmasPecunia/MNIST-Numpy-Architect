import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16*7*7, 10)
 
)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(model.parameters() ,lr=0.01)

EPOCHS = 50

for epochs in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)

    if epochs % 2 == 0:
        print(f"Epoch {epochs}: loss {avg_loss:.4f}")

model.eval()

correct = 0
total = 0

all_preds = []
all_true = []

with torch.no_grad():

    for xb, yb in test_loader:

        logits = model(xb)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == yb).sum().item()
        total += yb.size(0)

        all_preds.extend(preds.numpy())
        all_true.extend(yb.numpy())

accuracy = correct / total
print(f"\nAccuracy TEST: {accuracy:.2f}")
torch.save(model, "mnist_model_2_0.pth")
print("Model saved")





