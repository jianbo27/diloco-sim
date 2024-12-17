import torch
import torch.nn.functional as F
from diloco_sim import DilocoSimulator, DilocoSimulatorConfig
from nanogpt import GPTConfig, GPT
from data import TextDataset
import numpy as np
import argparse
from util import arg_combinations, str2bool, generate_text
import random
import torch.autograd.profiler as profiler


def CELoss(inputs, targets):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", "-b", type=int, nargs="+", default=32)
    parser.add_argument("--num_nodes", "-n", type=int, nargs="+", default=4)
    # parser.add_argument("--p_shuffle", "-p", type=float, nargs="+", default=0.01)
    parser.add_argument("--learning_rate", "-lr", type=float, nargs="+", default=0.001)
    parser.add_argument("--outer_learning_rate", type=float, nargs="+", default=0.7)
    parser.add_argument("--outer_momentum", type=float, nargs="+", default=0.9)
    parser.add_argument("--max_local_step", type=int, nargs="+", default=5000)
    parser.add_argument("--save_dir", type=str, nargs="+", default=None)
    parser.add_argument("--ckpt_interval", type=int, nargs="+", default=None)
    parser.add_argument("--model_path", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_project", type=str, nargs="+", default=None)
    parser.add_argument("--eval_iters", type=int, nargs="+", default=400)
    parser.add_argument("--diloco_interval", type=int, nargs="+", default=500)
    parser.add_argument("--cosine_anneal", type=str2bool, nargs="+", default=False)
    parser.add_argument("--warmup_steps", type=int, nargs="+", default=0)
    parser.add_argument(
        "--model_size", type=str, nargs="+", default="small", choices=["small", "base", "medium", "large", "xl"]
    )
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--dataset_path", type=str, default="data/owt/openwebtext.bin")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")

    base_args = parser.parse_args()

    for args in arg_combinations(base_args):

        print("Running with args:\n", args)

        if args.seed:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        gptconf = {
            "gpt2_small": GPTConfig.gpt2_small,
            "gpt2_base": GPTConfig.gpt2_base,
            "gpt2_medium": GPTConfig.gpt2_medium,
            "gpt2_large": GPTConfig.gpt2_large,
            "gpt2_xl": GPTConfig.gpt2_xl,
        }[args.model_size]()

        train_dataset = TextDataset(
            args.dataset_path,
            dtype=np.uint16,
            seq_length=gptconf.block_size,
        )

        diloco_config = DilocoSimulatorConfig(
            model_cls=GPT,
            model_kwargs={"config": gptconf},
            optimizer_kwargs={"lr": args.learning_rate},
            train_dataset=train_dataset,
            loss_fn=CELoss,
            num_epochs=1,
            batch_size=args.batch_size,
            num_nodes=args.num_nodes,
            max_local_step=args.max_local_step,
            save_dir=args.save_dir,
            ckpt_interval=args.ckpt_interval,
            model_path=args.model_path,
            wandb_project=args.wandb_project,
            eval_iters=args.eval_iters,
            diloco_interval=args.diloco_interval,
            outer_optimizer_kwargs={"lr": args.outer_learning_rate, "momentum": args.outer_momentum, "nesterov": True},
            cosine_anneal=args.cosine_anneal,
            warmup_steps=args.warmup_steps,
        )

        diloco_sim = DilocoSimulator(diloco_config)

        assert args.train ^ args.generate ^ args.profile, "Must specify exactly one of train, generate, profile"

        if args.model_path:
            diloco_sim.load_model(args.model_path)

        if args.train:
            diloco_sim.train()
        elif args.generate:
            generate_text(diloco_sim.master_model)
        elif args.profile:
            pass

            # cuda_is_available = torch.cuda.is_available()

            # print("Warmup")
            # for _ in range(5):
            #     diloco_sim._train_step()
            #     diloco_sim._shuffle_params()

            # if cuda_is_available:
            #     torch.cuda.synchronize()

            # print("Profiling shuffle_params")
            # with profiler.profile(use_cuda=cuda_is_available) as prof:
            #     diloco_sim._shuffle_params()
            #     # wm._train_step()

            # if cuda_is_available:
            #     torch.cuda.synchronize()

            # sort_by = "cuda_time_total" if cuda_is_available else "cpu_time_total"

            # print(prof.key_averages().table(sort_by=sort_by, row_limit=50))
