import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

# Training loop
def train(model, dataloader, optimizer, criterion, grad_clip, device, scheduler, epochs):
    train_losses = []
    val_losses = [] 
    scaler = GradScaler() # Mixed precision scaler 

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}') # Progress bar if needed

        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            
            with autocast(): # for mixed precision
                # Forward pass - computation and calculating loss function 
                logits = model(input_ids)

                # Reshape logits and targets for loss computation
                logits = logits.view(-1, vocab_size)
                targets = targets.view(-1)

                # Compute loss
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()

            if grad_clip: # to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Backward pass and optimization - computing gradient descent and updating weight
            loss.backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        scheduler.step()

        train_losses.append(avg_loss)
        val_loss = validate(model, dataloader, criterion, device) # can use evaluate.py
        val_losses.append(val_loss)
    torch.save(model.state_dict(), "model.pth")
    plot_graph(train_losses, val_losses)

def plot_graph(train_losses, val_losses, epochs): # for analyzing loss for each epochs (if needed)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation loss over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 5000  # Unique tokens from tokenizer
    embed_dim = 128 # Dimension of the token for positional embedding - (position of a token)
    num_heads = 4 # Number of attention heads - relationship between tokens 
    num_layers = 4 # Number of transformer blocks i.e number of layers 
    max_len = 256 # Max len of the token 
    dropout = 0.1 # used to prevent overfitting
    learning_rate = 5e-4
    grad_clip = 1.0
    batch_size = 32
    epochs = 50

    # Model initialization
    model = TamilLanguageModel(vocab_size, embed_dim, num_heads, num_layers, max_len, dropout) # have to get model from model.py 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0) # CrossEntropyLoss - difference between two probability distribution
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    tokenized_texts = [[1, 5, 9, 12, 4], [3, 8, 10, 7]] * 500 #(eg.) # have to get tokenized_texts from tokenizer
    dataset = TamilDataset(tokenized_texts, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # to update the learning_rate for every iteration

    train(model, dataloader, optimizer, criterion, grad_clip, device, scheduler, epochs = 5)

