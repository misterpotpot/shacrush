# asset.py

import torch
import hashlib
import random
import string
import argparse
from model import PasswordCharPredictor, predict_first_character
from constants import MODEL_PATH, ACCURACY_DEFAULT


def generate_random_ascii_string(min_length=8, max_length=24):
    length = random.randint(min_length, max_length)
    return ''.join(random.choice([chr(i) for i in range(32, 127)]) for _ in range(length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate accuracy of a PasswordCharPredictor model.')
    parser.add_argument('--num', type=int, default=ACCURACY_DEFAULT, help='Number of strings to test.')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Model path to assess.')
    parser.add_argument('--verbose', type=bool, default=False, help='If set to True, it prints the result of each prediction.')
    args = parser.parse_args()
    print(f"Assessment of the model: {args.model}")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PasswordCharPredictor().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Initialize counters
    correct_predictions = 0

    # Generate strings, predict, and calculate accuracy
    for i in range(args.num):
        random_string = generate_random_ascii_string()
        hash_obj = hashlib.sha256(random_string.encode())
        hash_str = hash_obj.hexdigest()

        # Predict the first character
        predicted_char = predict_first_character(hash_str, model=model)
        if args.verbose is True:
            print(f"\nSimulation: {i+1}")
            print(f"hash: {hash_str}")
            print(f"string: {random_string}")
            print(f"predicted_char: {predicted_char}")
            print(f"is_correct: {predicted_char == random_string[0]}")

        # Check if the prediction is correct
        if predicted_char == random_string[0]:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / args.num
    print(f"\n\nAccuracy: {accuracy * 100}%")