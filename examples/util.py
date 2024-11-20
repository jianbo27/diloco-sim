import torch
import torch.nn.functional as F
from itertools import combinations
import time
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


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        # print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result

    return wrapper


def cosine_similarity(models):
    if len(models) < 2:
        return 1.0

    similarities = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_similarities = []

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Compute cosine similarity between flattened parameters
            sim = F.cosine_similarity(param1_flat, param2_flat, dim=0)
            layer_similarities.append(sim.item())

        # Average similarity for this pair of models across all layers
        similarities.append(sum(layer_similarities) / len(layer_similarities))

    # Return the average similarity across all model pairs
    return sum(similarities) / len(similarities)


def euclidean_distance(models):
    if len(models) < 2:
        return 0.0

    distances = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_distances = []

        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 != name2:
                raise ValueError("Model layers do not match.")
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Compute Euclidean distance between flattened parameters
            dist = torch.norm(param1_flat - param2_flat, p=2)
            layer_distances.append(dist.item())

            # if dist.item() > 0.0001:
            #     print(f"Layer '{name1}' divergence: {dist.item()}")

        # Average distance for this pair of models across all layers
        distances.append(sum(layer_distances) / len(layer_distances))

    # Return the average distance across all model pairs
    return sum(distances) / len(distances)


def mean_squared_difference(models):
    if len(models) < 2:
        return 0.0

    differences = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_differences = []

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Compute mean squared difference
            diff = torch.mean((param1_flat - param2_flat) ** 2)
            layer_differences.append(diff.item())

        # Average difference for this pair of models across all layers
        differences.append(sum(layer_differences) / len(layer_differences))

    # Return the average difference across all model pairs
    return sum(differences) / len(differences)


import torch
from itertools import combinations


def parameter_correlation(models):
    if len(models) < 2:
        return 1.0

    correlations = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_correlations = []

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Check if the variance is zero to prevent NaN
            if param1_flat.var() == 0 or param2_flat.var() == 0:
                layer_correlations.append(1.0)  # Identical parameters have a correlation of 1
            else:
                # Compute Pearson correlation
                corr = torch.corrcoef(torch.stack([param1_flat, param2_flat]))[0, 1]
                layer_correlations.append(corr.item())

        # Average correlation for this pair of models across all layers
        correlations.append(sum(layer_correlations) / len(layer_correlations))

    # Return the average correlation across all model pairs
    return sum(correlations) / len(correlations)


def drift_penalty(model, ref_model, weight=0.01):
    penalty = 0.0
    for (name, param), (_, ref_param) in zip(model.named_parameters(), ref_model.named_parameters()):
        # Compute the L2 norm difference with the reference parameter
        penalty += torch.norm(param - ref_param) ** 2
    return weight * penalty


def generate_text(model):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    while True:
        user_input = input("Enter the start of your text (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break

        generated_ids = model.generate(
            torch.tensor(tokenizer.encode(user_input)).unsqueeze(0),
            max_new_tokens=500,
            temperature=0.7,
            top_k=None,
        )

        generated_text = tokenizer.decode(generated_ids.squeeze())
        print(f"Generated Text: {generated_text}")

    # according to gradient magnitude
    # def _shuffle_params(self):
    #     with torch.no_grad():
    #         model_params = [list(model.parameters()) for model in self.models]

    #         L = len(model_params[0])

    #         for param_idx in range(L):

    #             p_shuffle = (
    #                 self.p_shuffle
    #                 if not self.modulate_p_shuffle
    #                 else self.p_shuffle * (1 - param_idx / (L - 1))
    #             )

    #             params = torch.stack(
    #                 [model[param_idx].view(-1) for model in model_params]
    #             )
    #             size = params.shape[1]

    #             gradient_magnitudes = torch.stack(
    #                 [model[param_idx].grad.view(-1).abs() for model in model_params]
    #             ).sum(dim=0)

    #             masked_indices = torch.topk(gradient_magnitudes, int(p_shuffle * size))[
    #                 1
    #             ]
    #             permutation_tensor = torch.rand(size, self.num_workers).argsort(
    #                 dim=1
    #             )  # TODO: shuffle p is actually p (1-1/Num workers) since might share with self
    #             row_indices = permutation_tensor.T
    #             column_indices = (
    #                 torch.arange(params.shape[1])
    #                 .unsqueeze(0)
    #                 .expand(params.shape[0], -1)
    #             )

    #             # masked_indices = torch.nonzero(
    #             #     torch.rand(size) < p_shuffle, as_tuple=True
    #             # )[0]

    #             params[:, masked_indices] = params[row_indices, column_indices][
    #                 :, masked_indices
    #             ]

    #             for model_idx, updated_param in enumerate(params):
    #                 model_params[model_idx][param_idx].data.copy_(
    #                     updated_param.view_as(model_params[model_idx][param_idx])
