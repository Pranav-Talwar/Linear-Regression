import torch
import torch.nn as nn
import torch.optim as optim

# We'll create a "batch" of 10 data points
N = 10
# Each data point has 1 feature
D_in = 1
# The output for each data point is a single value
D_out = 1

# Create our input data X
# Shape: (10 rows, 1 column)
X = torch.randn(N, D_in)

# Create our true target labels y by applying the "true" function
# and adding some noise for realism
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 # Add a little noise

# Hyperparameters
learning_rate = 0.01
epochs = 100

# Inherit from nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        # Call the constructor of the parent class (nn.Module)
        super().__init__()
        # Define the single layer we will use
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Define the forward pass: just pass the input through our one layer
        return self.linear_layer(x)

# Instantiate the model
# D_in = 1 feature, D_out = 1 output
model = LinearRegressionModel(in_features=1, out_features=1)

# nn.Module automatically finds all the parameters for you!
print("Model Architecture:")
print(model)
print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# Create an Adam optimizer
# We pass model.parameters() to tell the optimizer which tensors it should manage.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# We also use a pre-built loss function from torch.nn
loss_fn = nn.MSELoss() # Mean Squared Error Loss

# The professional training loop
for epoch in range(epochs):
    ### FORWARD PASS ###
    # Use the model to make a prediction
    y_hat = model(X)

    ### CALCULATE LOSS ###
    # Use our pre-built loss function
    loss = loss_fn(y_hat, y_true)

    ### THE THREE-LINE MANTRA ###
    # 1. Zero the gradients from the previous iteration
    optimizer.zero_grad()
    # 2. Compute gradients for this iteration
    loss.backward()
    # 3. Update the parameters
    optimizer.step()

    # Optional: Print progress
    if epoch % 10 == 0:
        # We can access the learned parameters through the model object
        w_learned = model.linear_layer.weight.item()
        b_learned = model.linear_layer.bias.item()
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}, W={w_learned:.3f}, b={b_learned:.3f}")

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, ffn_dim):
        super().__init__()
        # In an LLM, embedding_dim might be 4096, ffn_dim might be 11000

        # We use the LEGO bricks we already know:
        self.layer1 = nn.Linear(embedding_dim, ffn_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(ffn_dim, embedding_dim)
        # We could add Dropout here as well

    def forward(self, x):
        # The data flow is exactly what you'd expect:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# You can now read and understand this code perfectly.
# It's a direct application of what we learned in Parts 8 & 9.
