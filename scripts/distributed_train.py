import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = torch.nn.Linear(784, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(5):
        for batch, (data, target) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(rank)
            target = target.to(rank)

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
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)