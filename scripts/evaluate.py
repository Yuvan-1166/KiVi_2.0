
import torch

from model import TamilLanguageModel


def generate_text(model, tokenizer, prompt, max_length, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    generated = input_ids
    for _ in range(max_length):
        logits = model(generated)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    return tokenizer.decode(generated[0].tolist())

if __name__ == "__main__":
    vocab_size = 30000  # Same as in training
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    max_len = 512

    model = TamilLanguageModel(vocab_size, embed_dim, num_heads, num_layers, max_len)
    model.load_state_dict(torch.load("tamil_model.pth"))
    model.eval()

    tokenizer = ...  # Load the same tokenizer used during training
    prompt = "தமிழ் மொழி"  # Example prompt
    generated_text = generate_text(model, tokenizer, prompt, max_length=100)
    print(generated_text)
