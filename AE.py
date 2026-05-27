import torch
import torch.nn as nn

# Fake data
X = torch.randn(10, 6)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Compress
        self.encoder = nn.Linear(6, 2)

        # Reconstruct
        self.decoder = nn.Linear(2, 6)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

model = AutoEncoder()

loss_fn = nn.MSELoss()

output = model(X)

# KEY DIFFERENCE:
loss = loss_fn(output, X)

print(loss)