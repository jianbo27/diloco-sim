import torch
from diloco_sim.sequential_diloco import SequentialDilocoSimulator, SequentialDilocoConfig
from transformers import AutoTokenizer, EsmForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import Dataset
import os

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]['input_ids']), torch.tensor(self.data[idx]['input_ids'])

class ModifiedSequentialDilocoSimulator(SequentialDilocoSimulator):
    def _eval_model(self):
        """Override evaluation to use loss instead of accuracy"""
        self.models[0].eval()
        total_loss = 0
        steps = 0

        with torch.no_grad():
            for x, y in self.eval_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.models[0](x)
                total_loss += loss.item()
                steps += 1
                
                if steps >= self.config.eval_iters:
                    break

        avg_loss = total_loss / steps
        print(f"Eval Loss: {avg_loss:.4f}")
        self.models[0].train()


class ModifiedSequentialDilocoSimulator(SequentialDilocoSimulator):
    def __init__(self, config: SequentialDilocoConfig) -> None:
        super().__init__(config)
        self.eval_losses = []
        self.eval_steps = []
        self.flops_count = 0

    def _count_flops_for_batch(self, model, batch_size):
        """Estimate FLOPs for ESM model forward pass"""
        total_flops = 0
        # Rough estimation based on parameter count
        total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        # Each parameter involved in forward and backward pass
        total_flops = total_params * batch_size * 2
        return total_flops

    def _eval_model(self):
        """Modified evaluation to track metrics"""
        self.models[0].eval()
        total_loss = 0
        steps = 0

        with torch.no_grad():
            for x, y in self.eval_dataloader:
                x = x.to(self.device)
                loss = self.models[0](x)
                total_loss += loss.item()
                steps += 1
                
                if steps >= self.config.eval_iters:
                    break

        avg_loss = total_loss / steps
        self.eval_losses.append(avg_loss)
        self.eval_steps.append(self.local_step)
        
        print(f"Eval Loss: {avg_loss:.4f}")
        self.models[0].train()
        return avg_loss

    def _train_step(self):
        x, _ = self._get_batch()
        
        # Update each model sequentially and count FLOPs
        for model, optimizer, scheduler in zip(
            self.models, self.optimizers, self.schedulers
        ):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Update FLOPs count
            self.flops_count += self._count_flops_for_batch(model, x.size(0))

    def train(self):
        start_time = time.time()
        self._setup()
        self._train_loop()
        
        if self.config.save_dir:
            self._save_checkpoint()
        
        # Print final stats
        print("\n" + "="*50)
        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Total FLOPs: {self.flops_count:,}")
        print(f"FLOPs per second: {self.flops_count/total_time:,.2f}")

        if torch.cuda.is_available():
            max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
            print(f"GPU memory: {max_gpu_memory:.2f} GB")
        print("=" * 50 + "\n")
        
        return self.eval_losses[-1] if self.eval_losses else None


if __name__ == "__main__":
    import time
    torch.manual_seed(12345)
    
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    raw_dataset = load_dataset("karinapikalyova/peptides")
    
    # Split dataset
    split_dataset = raw_dataset['train'].train_test_split(
        test_size=0.1,
        train_size=0.9,
        seed=42
    )
    
    # Preprocess function
    def preprocess_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    # Process datasets
    train_tokenized = split_dataset['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=split_dataset['train'].column_names
    )
    
    test_tokenized = split_dataset['test'].map(
        preprocess_function,
        batched=True,
        remove_columns=split_dataset['test'].column_names
    )
    
    # Create dataset objects
    train_dataset = TokenizedDataset(train_tokenized)
    test_dataset = TokenizedDataset(test_tokenized)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    class ESMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        def forward(self, x):
            # Create the masked inputs using the data collator
            batch = data_collator([{'input_ids': ids} for ids in x])
            outputs = self.model(**{k: v.to(x.device) for k, v in batch.items()})
            return outputs.loss
    
    # Create save directory if it doesn't exist
    save_dir = "./esm_checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create config with adjusted parameters for benchmarking
    config = SequentialDilocoConfig(
        model_cls=ESMModel,
        model_kwargs={},
        optimizer_kwargs={"lr": 2e-5, "weight_decay": 0.01},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=lambda x, y: x,
        num_epochs=3,  # Keep small for benchmarking
        num_nodes=4,
        batch_size=8,
        diloco_interval=100,  # More frequent evaluation for better curves
        eval_iters=10,  # Reduced for faster benchmarking
        cosine_anneal=True,
        save_dir=save_dir
    )
    
    simulator = ModifiedSequentialDilocoSimulator(config)
    simulator.train()