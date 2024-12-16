import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm


# Load the OpenWebText dataset from HuggingFace
def download_openwebtext():
    dataset = load_dataset("openwebtext", split="train")
    return dataset


# Tokenize the dataset and append the EOT token
def tokenize_dataset_with_eot(dataset, tokenizer, eot_token_id):
    def tokenize_function(example):
        tokenized = tokenizer(example["text"], truncation=True, padding=False)
        # Append EOT token ID to the end of each tokenized input
        tokenized["input_ids"].append(eot_token_id)
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=False, num_proc=cpu_count())
    return tokenized_dataset


# Save the tokenized dataset into a .bin file
def save_tokenized_to_bin(tokenized_dataset, output_file):
    # Use tqdm to display progress while iterating over the dataset
    with open(output_file, "wb") as f:
        for entry in tqdm(tokenized_dataset, desc="Saving to .bin file", unit="entries"):
            input_ids = entry["input_ids"]
            # Convert to NumPy array and save in binary format
            input_ids_np = np.array(input_ids, dtype=np.uint16)
            input_ids_np.tofile(f)


def main():
    print("Preparing OpenWebText dataset for training...")

    print("Step 1: Downloading OpenWebText dataset...")
    dataset = download_openwebtext()

    print("Step 2: Tokenizing the dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 tokenizer

    eot_token_id = tokenizer.eos_token_id  # Typically 50256 for GPT-2

    tokenized_dataset = tokenize_dataset_with_eot(dataset, tokenizer, eot_token_id)

    print("Step 3: Saving the tokenized dataset to a .bin file...")
    output_file = "openwebtext.bin"
    save_tokenized_to_bin(tokenized_dataset, output_file)
    print(f"Tokenized dataset with EOT saved to {output_file}")


if __name__ == "__main__":
    main()
