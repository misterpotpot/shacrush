# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PasswordCharPredictor, HashDataset
import argparse
from constants import DATASET_DIR, DATASET_NB, MODEL_PATH, EPOCH_DEFAULT, LEARNING_DEFAULT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch the training of the PasswordCharPredictor Model.')
    parser.add_argument('--num_epochs', type=int, default=EPOCH_DEFAULT, help='Number of epochs to train for.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_DEFAULT, help='Learning rate.')
    args = parser.parse_args()

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = PasswordCharPredictor().to(device)
    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded pre-trained model.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for i in range(1, DATASET_NB + 1):  # Loop through each dataset file
            file_path = os.path.join(DATASET_DIR, f'hash{i}.json')
            dataset = HashDataset(file_path)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Adjust batch size as needed

            for hashes, labels in dataloader:
                # Move data to the device
                hashes, labels = hashes.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(hashes)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Dsiplay loss and average loss
            item_loss = loss.item()
            epoch_loss += item_loss
            avg_loss = epoch_loss / i
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Dataset {i}, Loss: {item_loss}, Avg Loss for epoch: {avg_loss}')

    # Save the model after training
    torch.save(model.state_dict(), MODEL_PATH)