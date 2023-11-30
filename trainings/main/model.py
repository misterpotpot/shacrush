# model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import argparse
from constants import MODEL_PATH


class PasswordCharPredictor(nn.Module):
    def __init__(self):
        super(PasswordCharPredictor, self).__init__()
        self.fc1 = nn.Linear(256, 176)  # SHA-256 hash input size is 256 bits
        self.fc2 = nn.Linear(176, 176)  # Hidden layer of a 176 neurons 'One hidden layer is sufficient for the large majority of problems [...] the optimal size of the hidden layer is usually between the size of the input and size of the output layers'. Jeff Heaton, the author of Introduction to Neural Networks in Java.
        self.fc3 = nn.Linear(176, 96)  # Output size is 96 (there only 96 printable ASCII character possibilities)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, as this will be used with a softmax for classification
        return x


class HashDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the string and hash from the dataset
        original_string = self.data[idx]['string']
        hash_str = self.data[idx]['hash']

        # Convert the hash string to a tensor of floats
        hash_tensor = torch.tensor([int(bit) for bit in bin(int(hash_str, 16))[2:].zfill(256)], dtype=torch.uint8)

        # Convert the first character of the original string to a one-hot encoded vector
        # Assuming ASCII, and the first character is the target
        char_index = ord(original_string[0]) - ord(' ')  # Subtract ord(' ') to align the ASCII index with 0
        char_tensor = torch.zeros(128, dtype=torch.float32)
        char_tensor[char_index] = 1.0

        return hash_tensor, char_tensor
    

def predict_first_character(hash_str, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model if not provided
    if model is None:
        # Ensure the model is in evaluation mode
        model = PasswordCharPredictor()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Ensure the model is in evaluation mode
    model.eval()
    
    # Convert the hash string to a tensor of floats
    hash_tensor = torch.tensor([int(bit) for bit in bin(int(hash_str, 16))[2:].zfill(256)], dtype=torch.uint8).to(device)
    
    # Forward pass to get output
    output = model(hash_tensor)

    # Get the predicted character index
    _, predicted_idx = torch.max(output.data, 1)
    predicted_char = chr(predicted_idx + ord(' '))  # Add ord(' ') to get the ASCII character

    return predicted_char


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a first character of a sha-256 string.')
    parser.add_argument('--hash', type=str, default='5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', help='Hash from which to guess a character.')
    args = parser.parse_args()

    result = predict_first_character(hash_str=args.hash, model=None)
    print(f'hash: {args.hash}')
    print(f'predicted_char: {result}')