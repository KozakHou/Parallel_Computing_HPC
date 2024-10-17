import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Wrap the model with DataParallel to enable multi-GPU training
model = nn.DataParallel(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to the available GPUs
model.cuda()

#   ... training ...