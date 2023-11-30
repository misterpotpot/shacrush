# SHA-256 First Character Predictor

## Objective

The objective of this project is to investigate whether a neural networks based model can reverse engineer the SHA-256 hashing algorithm.
Initially, this neural network is designed to predict the first character of an 8 to 24 characters ASCII string based on its SHA-256 hash.
This serves as a Proof of Concept, which we could expand upon if it looks promising.
The ultimate goal is to reverse engineer complete ASCII strings of known lengths from their SHA-256 hashes, employing individual neural networks for each character. Subsequently, we aspire to adapt this methodology then to tackle any inputs of unknown lengths.

Key Points for Assessment:

- Does the average accuracy of the produced models significantly exceed 1/128 (given that there are 128 characters in the ASCII set)?
- Does the loss function of the produced models show significant progress within a training epoch?

The results obtained will then serve as the baseline for calculations to determine if this method can effectively reverse engineer SHA-2 algorithms, such as SHA-256, in real-world conditions.

## Key results

Work In Progress (WIP)

## Usage

### Dataset Preparation

To train your custom model, you need a dataset. This dataset is located in the `dataset` folder and consists of a json file named `hash.json`.

You have two options to generate those `hash.json` files :

1. Download Pre-structured Data:

   - Obtain JSON files containing data in an array with two keys: `string` and `hash`.

2. Generate Data:
   - Utilize the `generate_hash.py` script to automatically populate the dataset folder for you. It accepts the following arguments:
     - `--num`: Number of files to generate (default: 4096)
     - `--size`: Number of hashes per file (default: 4096)
     - `--sufix`: If you have generated files before, you can continue the numbering from the next number (default: 1)

You can proceed to the next steps once you have your `dataset` folder ready with hash JSON files named as follows `hash1.json`, `hash2.json`...

### Model Setup

For each model or setup, it is recommended to create a dedicated folder. Begin by copying the `main` folder, then modify the contents of the newly created directory as desired. The `main` folder includes default parameters and neural layer configurations, along with a pre-trained default model using 16.7 million hashes, which is suitable for testing purposes

### Training

Before training your model with `train.py`, make sure to check and update the constants of your subfolder in the `constants.py` file to suit your setup.

Pay special attention to these variables:

- `DATASET_NB`: The number of JSON files in the dataset folder.
- `MODEL_PATH`: The name of the model file when it will be saved or reused.
