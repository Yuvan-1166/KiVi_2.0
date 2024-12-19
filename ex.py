import torch

from model import BART_model
from tokenizer import Tokenizer


def chat():
    model_path = "path_to_your_model"  # Path to your trained BART model
    model = BART_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tokens = Tokenizer()
    conversation_history = []
    while True:
        # Create a string from the conversation history
        history_string = "\n".join(conversation_history)
        
        # Get user input
        input_text = input("> ")
        if(input_text == "> q"):
            print("Quitting chat...")
            break
        
        # Append user input to conversation history
        conversation_history.append(f":User  {input_text}")
        
        # Encode the input and history
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate a response
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Append the response to conversation history
        conversation_history.append(f"Bot: {response}")
        
        # Print the bot's response
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
