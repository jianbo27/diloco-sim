import argparse
import itertools
import copy
from transformers import AutoTokenizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def arg_combinations(args):
    # Identify arguments that are lists of values and should be iterated over
    args_dict = vars(args)
    list_args = {key: value for key, value in args_dict.items() if isinstance(value, list)}

    # Identify static arguments (those with a single value)
    static_args = {key: value for key, value in args_dict.items() if key not in list_args}

    if not list_args:
        yield argparse.Namespace(**static_args)
        return

    # Generate all combinations of list arguments
    keys, values = zip(*list_args.items())
    for combination in itertools.product(*values):
        # Create a new Namespace with the static arguments
        combined_args = copy.deepcopy(static_args)

        # Add the current combination of list arguments
        combined_args.update(dict(zip(keys, combination)))

        # Yield a Namespace object with the combined arguments
        yield argparse.Namespace(**combined_args)


def diloco_args_generator(extra_args={}):
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", "-b", type=int, nargs="+", default=32)
    parser.add_argument("--num_workers", "-w", type=int, nargs="+", default=4)
    parser.add_argument("--learning_rate", "-lr", type=float, nargs="+", default=0.001)
    parser.add_argument("--outer_learning_rate", type=float, nargs="+", default=0.7)
    parser.add_argument("--max_local_step", type=int, nargs="+", default=5000)
    parser.add_argument("--save_dir", type=str, nargs="+", default=None)
    parser.add_argument("--ckpt_interval", type=int, nargs="+", default=None)
    parser.add_argument("--model_path", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_project", type=str, nargs="+", default=None)
    parser.add_argument("--eval_iters", type=int, nargs="+", default=100)
    parser.add_argument("--diloco_interval", type=int, nargs="+", default=None)
    parser.add_argument("--cosine_anneal", type=str2bool, nargs="+", default=False)
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--log_stats_interval", type=int, nargs="+", default=10)
    parser.add_argument("--device", type=str, nargs="+", default=None)
    parser.add_argument("--compile", action="store_true")

    for arg, arg_type in extra_args:
        parser.add_argument(f"--{arg}", type=arg_type, nargs="+", default=None)

    base_args = parser.parse_args()

    for arg_combo in arg_combinations(base_args):
        yield {
            "batch_size": arg_combo.batch_size,
            "num_workers": arg_combo.num_workers,
            "learning_rate": arg_combo.learning_rate,
            "outer_learning_rate": arg_combo.outer_learning_rate,
            "max_local_step": arg_combo.max_local_step,
            "save_dir": arg_combo.save_dir,
            "ckpt_interval": arg_combo.ckpt_interval,
            "model_path": arg_combo.model_path,
            "wandb_project": arg_combo.wandb_project,
            "eval_iters": arg_combo.eval_iters,
            "diloco_interval": arg_combo.diloco_interval,
            "cosine_anneal": arg_combo.cosine_anneal,
            "seed": arg_combo.seed,
            "log_stats_interval": arg_combo.log_stats_interval,
            "device": arg_combo.device,
            "compile": arg_combo.compile,
            **{arg: arg_combo[arg] for arg, arg_type in extra_args},
        }
