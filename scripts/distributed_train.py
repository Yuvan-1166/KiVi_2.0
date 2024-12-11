import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import torch.distributed as dist
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.distributed.is_available())

def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = '192.168.114.65'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, backend):
    setup(rank, world_size, backend)

    # Use GPU if available, otherwise fallback to CPU
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() and backend == "nccl" else 'cpu')

    # Data transformation
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Model setup
    model = torch.nn.Linear(784, 10).to(device)
    ddp_model = DDP(model, device_ids=[rank] if backend == "nccl" else None)

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        for batch, (data, target) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(device)  # Flatten MNIST images
            target = target.to(device)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch}, Loss {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Select the backend: "nccl" for GPU, "gloo" for CPU
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Run distributed training
    torch.multiprocessing.spawn(train, args=(world_size, backend), nprocs=world_size, join=True)
