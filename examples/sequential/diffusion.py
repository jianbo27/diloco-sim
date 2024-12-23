import torch
from diloco_sim.sequential_diloco import SequentialDilocoSimulator, SequentialDilocoConfig
from diffusers import UNet2DModel, DDPMScheduler
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
import os

class ImageDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("fashion_mnist")[split]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        x = self.transform(image)
        x = x.reshape(1, 28, 28)
        return x, x

class DiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=32,
            in_channels=1,
            out_channels=1,
            center_input_sample=False,
            time_embedding_type="positional",
            freq_shift=0,
            flip_sin_to_cos=True,
            down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D"],
            up_block_types=["UpBlock2D", "UpBlock2D", "UpBlock2D"],
            block_out_channels=(64, 128, 256),
            layers_per_block=2,
        )
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        padded_x = torch.nn.functional.pad(x, (2, 2, 2, 2))
        
        noise = torch.randn_like(padded_x)
        timesteps = torch.randint(0, 1000, (batch_size,), device=x.device)
        
        noisy_images = self.noise_scheduler.add_noise(padded_x, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps).sample
        
        return torch.nn.functional.mse_loss(noise_pred, noise)

class ModifiedSequentialDilocoSimulator(SequentialDilocoSimulator):
    def _eval_model(self):
        """Override evaluation to use MSE loss instead of accuracy"""
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
        return avg_loss

if __name__ == "__main__":
    save_dir = "./diffusion_diloco_checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_dataset = ImageDataset("train")
    test_dataset = ImageDataset("test")
    
    config = SequentialDilocoConfig(
        model_cls=DiffusionModel,
        model_kwargs={},
        optimizer_kwargs={
            "lr": 1e-4,
            "weight_decay": 1e-6
        },
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=lambda x, y: x,
        num_epochs=100,
        num_nodes=4,
        batch_size=16,
        diloco_interval=100,
        eval_iters=10,
        cosine_anneal=True,
        save_dir=save_dir
    )
    
    # Use our modified simulator instead of the original
    simulator = ModifiedSequentialDilocoSimulator(config)
    simulator.train()