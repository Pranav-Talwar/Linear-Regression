import torch

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

# Create a Linear layer
linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)

# You can inspect the randomly initialized parameters inside
print(f"Layer's Weight (W): {linear_layer.weight}\n")
print(f"Layer's Bias (b): {linear_layer.bias}\n")

# You use it just like a function. Let's pass our data X through it.
# This performs the forward pass: X @ W.T + b
# (Note: nn.Linear stores W as (D_out, D_in), so it uses a transpose)
y_hat_nn = linear_layer(X)

print(f"Output of nn.Linear (first 3 rows):\n {y_hat_nn[:3]}")

# Create a ReLU activation function layer
relu = torch.nn.ReLU()

# Let's create some sample data with positive and negative values
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

# Apply the activation
activated_data = relu(sample_data)

print(f"Original Data:      {sample_data}")
print(f"Data after ReLU:    {activated_data}")

# Create a GELU activation function layer
gelu = torch.nn.GELU()

# Use the same sample data
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

# Apply the activation
activated_data = gelu(sample_data)

print(f"Original Data:      {sample_data}")
print(f"Data after GELU:    {activated_data}")

# Softmax must be told which dimension contains the scores to be normalized.
# For a batch of predictions, this is typically the last dimension (dim=-1 or dim=1).
softmax = torch.nn.Softmax(dim=-1)

# Let's create some sample logits for a batch of 2 items, with 4 possible classes
# A higher number means the model is more confident in that class.
logits = torch.tensor([
    [1.0, 3.0, 0.5, 1.5],  # Logits for item 1
    [-1.0, 2.0, 1.0, 0.0]   # Logits for item 2
])

# Apply the softmax
probabilities = softmax(logits)

print(f"Original Logits:\n {logits}\n")
print(f"Output Probabilities:\n {probabilities}\n")
print(f"Sum of probabilities for item 1: {probabilities[0].sum()}")
print(f"Sum of probabilities for item 2: {probabilities[1].sum()}")

# --- nn.Embedding ---
# Let's define a small vocabulary and embedding size
vocab_size = 10       # We have 10 unique words in our language
embedding_dim = 3     # Each word will be represented by a vector of size 3

# Create the embedding layer
embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Input: A batch of tokenized sentences. Let's make a batch of 1 sentence with 4 words.
# The numbers are the integer IDs for each word in our vocabulary.
input_ids = torch.tensor([[1, 5, 0, 8]]) # Shape: (batch_size=1, sequence_length=4)

# Get the vectors for our sentence
word_vectors = embedding_layer(input_ids)

print(f"Embedding Layer's Weight Matrix (shape {embedding_layer.weight.shape}):\n {embedding_layer.weight.data}\n")
print(f"Input Token IDs (shape {input_ids.shape}):\n {input_ids}\n")
print(f"Output Word Vectors (shape {word_vectors.shape}):\n {word_vectors}")

# --- nn.LayerNorm ---
# LayerNorm needs to know the shape of the features it's normalizing.
# Our word vectors from the embedding example have a feature dimension of 3.
feature_dim = 3
norm_layer = torch.nn.LayerNorm(normalized_shape=feature_dim)

# Input: A batch of feature vectors. Let's create one.
input_features = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]) # Shape (1, 2, 3)

# Apply the normalization
normalized_features = norm_layer(input_features)

# Let's check the mean and standard deviation of the output
output_mean = normalized_features.mean(dim=-1)
output_std = normalized_features.std(dim=-1)

print(f"Input Features:\n {input_features}\n")
print(f"Normalized Features:\n {normalized_features}\n")
print(f"Mean of each output vector (should be ~0): {output_mean}")
print(f"Std Dev of each output vector (should be ~1): {output_std}")

# --- nn.Dropout ---
# Create a dropout layer that will zero out 50% of its inputs
dropout_layer = torch.nn.Dropout(p=0.5)

# Input: A simple tensor of ones so we can see the effect clearly.
input_tensor = torch.ones(1, 10) # Shape (1, 10)

# IMPORTANT: Dropout is only active during training. You must tell the layer
# it's in training mode with .train() or evaluation mode with .eval().
dropout_layer.train() # Activate dropout
output_during_train = dropout_layer(input_tensor)

dropout_layer.eval() # Deactivate dropout
output_during_eval = dropout_layer(input_tensor)

print(f"Input Tensor:\n {input_tensor}\n")
print(f"Output during training (randomly zeroed and scaled):\n {output_during_train}\n")
print(f"Output during evaluation (identity function):\n {output_during_eval}")
