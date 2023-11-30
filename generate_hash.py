# generate_hash.py

import json
import random
import hashlib
import argparse


def generate_random_ascii_string(min_length=8, max_length=24):
    length = random.randint(min_length, max_length)
    return ''.join(random.choice([chr(i) for i in range(32, 127)]) for _ in range(length))


def generate_hashed_data(num_strings):
    data = []
    for _ in range(num_strings):
        random_string = generate_random_ascii_string()
        hash_object = hashlib.sha256(random_string.encode())
        hashed_string = hash_object.hexdigest()
        data.append({"string": random_string, "hash": hashed_string})
    return data


def create_datasets(num_datasets, dataset_size, file_suffix):
    for i in range(num_datasets):
        hashed_data = generate_hashed_data(dataset_size)
        file_name = f'./dataset/hash{file_suffix + i}.json'
        with open(file_name, 'w') as file:
            json.dump(hashed_data, file)
        print(f'Dataset {file_suffix + i} created.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hashed datasets.')
    parser.add_argument('--num', type=int, default=4096, help='Number of datasets to create (default: 1)')
    parser.add_argument('--suffix', type=int, default=1, help='Starting suffix for file names (default: 1)')
    parser.add_argument('--size', type=int, default=4096, help='Size of each dataset (default: 4096)')
    args = parser.parse_args()

    create_datasets(num_datasets=args.num, dataset_size=args.size, file_suffix=args.suffix)