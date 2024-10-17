from torch.cuda.amp import autocast, GradScaler


model = ...
criterion = ...
dataloader = ...
optimizer = ...

# Initialize GradScaler for mixed precision
scaler = GradScaler()

for inputs, labels in dataloader:
    inputs, labels = inputs.cuda(), labels.cuda()

    optimizer.zero_grad()

    # Perform forward pass and backward pass with mixed precision
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    # Scale gradients and perform optimizer step
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()