import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# Data
x_train = np.linspace(-1, 1, 200).reshape(-1, 1).astype(np.float32)
y_train = runge(x_train)

x_val = np.linspace(-1, 1, 50).reshape(-1, 1).astype(np.float32)
y_val = runge(x_val)

x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)
x_val_tensor = torch.tensor(x_val)
y_val_tensor = torch.tensor(y_val)

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
train_losses, val_losses = [], []
epochs = 3000
for epoch in range(epochs):
    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_val_tensor)
        val_loss = criterion(y_val_pred, y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    if (epoch + 1) % 200 == 0:
        print(f"Epoch:[{epoch + 1}/3000], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

# Prediction
model.eval()
with torch.no_grad():
    y_pred_tensor = model(x_val_tensor)
y_pred = y_pred_tensor.numpy()

# Error metrics
mse = np.mean((y_val - y_pred)**2)
max_err = np.max(np.abs(y_val - y_pred))
print()
print("Validation MSE:", mse, "Max error:", max_err)

# Plot function vs NN
plt.figure()
plt.plot(x_val, y_val, label="True Runge Function", color="blue")
plt.plot(x_val, y_pred, label="NN Prediction (MSE)", color="orange", linestyle="--")
plt.legend()
plt.title("Runge Function Approximation (MSE)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# Plot training/validation loss
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss Curves (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
