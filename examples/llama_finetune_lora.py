import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from diloco_sim import DilocoSimulator, DilocoSimulatorConfig
import json
import os
import torch.nn as nn
import wandb
from tqdm import tqdm

def identity_loss(x, _):
    """Identity loss function that returns the input as is"""
    return x

def get_lora_params(model):
    """Helper function to get only LoRA parameters"""
    for name, param in model.named_parameters():
        if 'lora_' in name:
            yield param


class LoraLlamaAlpacaModel(nn.Module):
    def __init__(
        self, 
        model_name: str, 
        use_bfloat16: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True
        )
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                param.data = param.data.to(torch.float32)
            else:
                param.requires_grad = False
                
    def forward(self, x):
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        labels = x[:, 2, :].long()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss.view(1)

class AlpacaDatasetProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_instruction(self, instruction: str, input_text: str = "", output: str = "") -> str:
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt
    
    def process_function(self, examples):
        prompts = [
            self.format_instruction(
                instruction=instr,
                input_text=inp if inp else "",
                output=out
            )
            for instr, inp, out in zip(examples['instruction'], examples['input'], examples['output'])
        ]
        
        encodings = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        return encodings


class AlpacaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        labels = self.encodings['labels'][idx]
        
        model_inputs = torch.stack([input_ids, attention_mask, labels])
        target = torch.tensor([0.0])
        return model_inputs, target

    def __len__(self):
        return len(self.encodings['input_ids'])


def load_alpaca_dataset():
    local_path = "alpaca_data.json"
    
    if not os.path.exists(local_path):
        print("Downloading Alpaca dataset...")
        import requests
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        response = requests.get(url)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(local_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset_dict = {
        'instruction': [],
        'input': [],
        'output': []
    }
    
    for item in data:
        dataset_dict['instruction'].append(item['instruction'])
        dataset_dict['input'].append(item['input'])
        dataset_dict['output'].append(item['output'])
    
    return Dataset.from_dict(dataset_dict)


def prepare_datasets(tokenizer, max_length: int = 512):
    dataset = load_alpaca_dataset()
    dataset = dataset.train_test_split(test_size=0.1)
    
    processor = AlpacaDatasetProcessor(tokenizer, max_length)
    
    train_encodings = processor.process_function(dataset["train"])
    eval_encodings = processor.process_function(dataset["test"])
    
    train_dataset = AlpacaDataset(train_encodings)
    eval_dataset = AlpacaDataset(eval_encodings)
    
    return train_dataset, eval_dataset


class WandBMonitor(DilocoSimulator):
    def _setup_master_model(self):
        """Override to only handle LoRA parameters in master model"""
        if self.rank == 0:
            self.master_model = self.model.__class__(**self.config.model_kwargs)
            for name, param in self.master_model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def _eval_model(self):
        """Override eval model to handle loss only"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for _ in range(self.config.eval_iters):
                x, y = self._get_batch(eval=True)
                output = self.model(x)
                total_loss += output.item()
                
                if self.rank == 0:
                    wandb.log({"eval_loss": output.item()})
                    
        avg_loss = total_loss / self.config.eval_iters
        if self.rank == 0:
            print(f"Eval Loss: {avg_loss}")
            
        self.model.train()
        return avg_loss

    def _outer_step(self):
        """Override outer step to handle LoRA parameters"""
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param_clone = param.data.clone().float()
                dist.all_reduce(param_clone)
                param_clone /= self.config.num_nodes
                param.data.copy_(param_clone)

        if self.rank == 0:
            self.master_optimizer.zero_grad()
            
            for name, param in self.model.named_parameters():
                if 'lora_' in name:
                    master_param = self.master_model.state_dict()[name]
                    master_param.grad = master_param.data - param.data.to(master_param.device)
                    
            self.master_optimizer.step()
            
            for name, param in self.master_model.named_parameters():
                if 'lora_' in name:
                    self.model.state_dict()[name].copy_(param.data)

        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                dist.broadcast(param.data, src=0)

    def _train_step(self):
        x, y = self._get_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = output
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.rank == 0:
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "step": self.local_step
            })

    def _train(self, rank: int):
        if rank == 0:
            wandb.init(
                project="llama-alpaca-lora",
                config={
                    "learning_rate": self.config.optimizer_kwargs["lr"],
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "num_nodes": self.config.num_nodes,
                    "model_name": self.config.model_kwargs["model_name"]
                }
            )
        
        try:
            super()._train(rank)
        finally:
            if rank == 0:
                wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, eval_dataset = prepare_datasets(tokenizer, max_length=512)

    lora_save_dir = "./llama_alpaca_lora_finetuned"
    os.makedirs(lora_save_dir, exist_ok=True)
    
    config = DilocoSimulatorConfig(
        model_cls=LoraLlamaAlpacaModel,
        model_kwargs={
            "model_name": model_name,
            "use_bfloat16": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        },
        optimizer_kwargs={
            "lr": 2e-4,
            "weight_decay": 0.01
        },
        loss_fn=identity_loss,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=1,
        num_epochs=3,
        save_dir=lora_save_dir,
        eval_iters=50,
        num_nodes=2,
        diloco_interval=500
    )
    
    trainer = WandBMonitor(config)
    trainer.train()