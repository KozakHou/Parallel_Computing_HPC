# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Create the dataset and data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Define the model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).cuda()

    # Wrap the model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()