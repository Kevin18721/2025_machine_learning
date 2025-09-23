import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Runge function and derivative
def runge(x):
    return 1 / (1 + 25 * x**2)

def runge_derivative(x):
    return -50 * x / (1 + 25 * x**2)**2

# Data
x_train = np.linspace(-1, 1, 200).reshape(-1, 1).astype(np.float32)
y_train = runge(x_train)
dy_train = runge_derivative(x_train)

x_val = np.linspace(-1, 1, 50).reshape(-1, 1).astype(np.float32)
y_val = runge(x_val)
dy_val = runge_derivative(x_val)

x_train_tensor = torch.tensor(x_train, requires_grad=True)
y_train_tensor = torch.tensor(y_train)
dy_train_tensor = torch.tensor(dy_train)

x_val_tensor = torch.tensor(x_val, requires_grad=True)
y_val_tensor = torch.tensor(y_val)
dy_val_tensor = torch.tensor(dy_val)

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
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 3000
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x_train_tensor)
    
    # Compute derivative
    dy_pred = torch.autograd.grad(
        outputs=y_pred,
        inputs=x_train_tensor,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]
    
    # Loss = f(x) MSE + f'(x) MSE
    loss = nn.MSELoss()(y_pred, y_train_tensor) + nn.MSELoss()(dy_pred, dy_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    x_val_tensor.requires_grad_(True)
    y_val_pred = model(x_val_tensor)
    dy_val_pred = torch.autograd.grad(
        outputs=y_val_pred,
        inputs=x_val_tensor,
        grad_outputs=torch.ones_like(y_val_pred),
        create_graph=False
    )[0]
    val_loss = nn.MSELoss()(y_val_pred, y_val_tensor) + nn.MSELoss()(dy_val_pred, dy_val_tensor)
    
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

# Prediction
model.eval()
x_val_tensor.requires_grad_(True)
y_pred = model(x_val_tensor)
dy_pred = torch.autograd.grad(
    outputs=y_pred,
    inputs=x_val_tensor,
    grad_outputs=torch.ones_like(y_pred),
    create_graph=False
)[0]

y_pred = y_pred.detach().numpy()
dy_pred = dy_pred.detach().numpy()

# Errors
mse_val = np.mean((y_val - y_pred)**2)
max_err_val = np.max(np.abs(y_val - y_pred))
mse_deriv = np.mean((dy_val - dy_pred)**2)
max_err_deriv = np.max(np.abs(dy_val - dy_pred))
print("Validation f(x) - MSE:", mse_val, "Max error:", max_err_val)
print("Validation f'(x) - MSE:", mse_deriv, "Max error:", max_err_deriv)

# Plot function
plt.figure()
plt.plot(x_val, y_val, label="True f(x)")
plt.plot(x_val, y_pred, '--', label="NN f(x)")
plt.legend()
plt.title("Function Approximation")
plt.show()

# Plot derivative
plt.figure()
plt.plot(x_val, dy_val, label="True f'(x)")
plt.plot(x_val, dy_pred, '--', label="NN f'(x)")
plt.legend()
plt.title("Derivative Approximation")
plt.show()

# Plot training/validation loss
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
