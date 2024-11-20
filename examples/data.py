import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import math


class TextDataset(Dataset):
    def __init__(self, bin_file_path, dtype=np.int32, seq_length=1024):
        """
        Args:
            bin_file_path (str): Path to the .bin file.
            dtype (type): Data type of the tokenized data (default: np.int32).
            seq_length (int): The fixed length of each sequence (x).
        """
        self.bin_file_path = bin_file_path
        self.dtype = dtype
        self.seq_length = seq_length

        # Create a memmap object for the entire binary file
        self.data = np.memmap(self.bin_file_path, dtype=self.dtype, mode="r")

        # Compute the total number of tokens in the dataset
        self.num_tokens = len(self.data)

        # Calculate how many sequences we can extract given the context length
        self.num_sequences = math.floor(self.num_tokens / (self.seq_length + 1))

    def __len__(self):
        # Return the number of sequences available based on the fixed seq_length
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Get the sequence at index idx.
        Returns the token IDs (x) and the next token (y) as a torch tensor.
        """
        start_idx = idx * (self.seq_length + 1)
        end_idx = start_idx + (self.seq_length + 1)
        sequence = self.data[start_idx:end_idx].astype(np.int32)

        x = torch.tensor(sequence[:-1], dtype=torch.long)  # Input sequence
        y = torch.tensor(sequence[1:], dtype=torch.long)

        return x, y


def stream_batches(dataset, batch_size, sequence_size, tokenizer):
    buffer = []
    for example in dataset:
        # Tokenize text and convert to input IDs
        tokens = tokenizer(example["text"], truncation=True, max_length=sequence_size)["input_ids"]

        # Add tokens to buffer
        buffer.extend(tokens)

        # Process buffer into chunks of (sequence_size)
        while len(buffer) >= sequence_size:
            # Take the first sequence_size tokens and reshape
            sequence = buffer[:sequence_size]
            buffer = buffer[sequence_size:]

            # Yield batch when we have enough sequences
            if len(sequence) == sequence_size:
                yield np.array(sequence).reshape(batch_size, sequence_size)
                sequence = []


# Example usage
if __name__ == "__main__":
    bin_file_path = "data/owt/openwebtext.bin"
    dataset = TextDataset(bin_file_path, seq_length=1024, dtype=np.uint16)  # Fixed context length of 1024 tokens

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Accessing the first (x, y) pair in the dataset
    i = 0

    while True:
        x, y = dataset[i]
        print(f"First sequence (x): {x}")
        print(f"Next token (y): {y}")
        print(f"len(x): {len(x)}, len(y): {len(y)}")

        num_eos_tokens = (x == tokenizer.eos_token_id).sum()
        print(f"Number of EOS tokens in x: {num_eos_tokens}")

        x_translated = tokenizer.decode(x)
        y_translated = tokenizer.decode(y)
        # print(f"First sequence (x): {x_translated}")
        # print(f"Next token (y): {y_translated}")
        i += 1

        input()
