import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import TamilLanguageModel


class TamilDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.data = tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        return torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)

def train(model, dataloader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch[:, :-1])
            loss = criterion(logits.view(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    file_path = "tamil_dataset.txt"
    vocab_size = 30000  # Adjust based on tokenizer
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    max_len = 512
    batch_size = 32
    learning_rate = 3e-4

    tokenizer = ...  # Define or load your Tamil tokenizer
    dataset = TamilDataset(file_path, tokenizer, block_size=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TamilLanguageModel(vocab_size, embed_dim, num_heads, num_layers, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, criterion, device)
    torch.save(model.state_dict(), "tamil_model.pth")

