import numpy as np
import torch
from datetime import datetime
from config import Config
from dataset import dataset_factory, load_all_data
from model import BrainNetworkTransformer
from components import optimizers_factory, lr_scheduler_factory, logger_factory
from train import Train
from sklearn.model_selection import StratifiedKFold
import itertools

def model_training(cfg, device, train_idx=None, test_idx=None):
    """Train a single fold (supports cross-validation indices)"""
    cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
    dataloaders = dataset_factory(cfg, train_idx=train_idx, test_idx=test_idx)
    logger = logger_factory(cfg)

    # Initialize model, optimizer and learning rate scheduler
    model = BrainNetworkTransformer(cfg).to(device)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)

    # Train and return the best results
    training = Train(cfg, model, optimizers, lr_schedulers, dataloaders, logger, device)
    return training.train()


def run_cv(cfg, device):
    """Execute 10-fold stratified cross-validation and return mean/std results"""
    # Load full dataset and get stratification label (site)
    time_series, _, _, labels, site, _, _ = load_all_data(cfg)

    # Initialize stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    all_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(time_series.numpy(), site.numpy())):
        print(f'\n===== Fold {fold_idx + 1}/{cfg.n_folds} =====')
        # Set fold-specific seed for reproducibility
        fold_seed = cfg.seed + fold_idx
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)

        # Train current fold
        fold_result = model_training(cfg, device, train_idx=train_idx, test_idx=test_idx)
        all_results.append(fold_result)
        print(f'Fold {fold_idx + 1} Completed | Test Accuracy: {fold_result["test_accuracy"]:.4f}')

    # Calculate mean and standard deviation across all folds
    if all_results:
        metrics = [
            "train_accuracy", "test_accuracy", "test_auc",
            "test_sensitivity", "test_specificity",
            "micro_f1", "micro_recall", "micro_precision"
        ]
        result_dict = {metric: [res[metric] for res in all_results] for metric in metrics}
        mean_dict = {metric: np.mean(result_dict[metric]) for metric in metrics}
        std_dict = {metric: np.std(result_dict[metric]) for metric in metrics}
        return mean_dict, std_dict
    else:
        return None, None

# ===================== Result Saving Functions =====================
def init_result_file(output_file="results.txt"):
    """Initialize result file with header (overwrite existing file)"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("Multi-Parameter Grid Search Results\n")
        f.write("=" * 120 + "\n")
        header = (f"{'DivW':<6} {'Layers':<6} {'Heads':<6} {'TopK':<6} {'LocalN':<6} "
                  f"{'Test Acc (Mean±Std)':<25} {'Test AUC (Mean±Std)':<25} {'F1 (Mean±Std)':<25}\n")
        f.write(header)
        f.write("-" * 120 + "\n")

def append_single_result(params, mean, std, output_file="results.txt"):
    """Append results of a single parameter combination to file"""
    div_w, layers, heads, topk, local_n = params
    with open(output_file, 'a', encoding='utf-8') as f:
        # Write concise results
        f.write(f"{div_w:<6.1f} {layers:<6d} {heads:<6d} {topk:<6.2f} {local_n:<6d} "
                f"{mean['test_accuracy']:.4f}±{std['test_accuracy']:.4f}   "
                f"{mean['test_auc']:.4f}±{std['test_auc']:.4f}   "
                f"{mean['micro_f1']:.4f}±{std['micro_f1']:.4f}\n")
        # Write detailed metrics
        f.write(f"  - Parameters: DivWeight={div_w:.1f}, Layers={layers}, Heads={heads}, TopK={topk:.2f}, LocalNeighbor={local_n}\n")
        for metric in mean.keys():
            f.write(f"    {metric:<15}: {mean[metric]:.7f} ± {std[metric]:.7f}\n")
        f.write("-" * 120 + "\n")

def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ===================== Grid Search Parameters =====================
    divergence_weights = [0.1]
    num_layers_list = [1]
    nhead_list = [4]
    global_topk_ratios = [0.1]
    local_neighbor_nums = [17]

    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        divergence_weights,
        num_layers_list,
        nhead_list,
        global_topk_ratios,
        local_neighbor_nums
    ))
    total_combinations = len(param_combinations)

    # Initialize result file
    init_result_file()
    print(f"\n===== Total Parameter Combinations: {total_combinations} | Results saved to results.txt =====")

    for idx, params in enumerate(param_combinations):
        div_w, layers, heads, topk_ratio, local_n = params

        # Update configuration dynamically
        cfg.divergence_weight = div_w
        cfg.model.num_layers = layers
        cfg.model.nhead = heads
        cfg.model.global_topk_ratio = topk_ratio
        cfg.model.local_neighbor_num = local_n

        # Print current training parameters
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total_combinations}] Training Parameters:")
        print(f"DivWeight: {div_w:.1f} | Layers: {layers} | Heads: {heads}")
        print(f"Global TopK: {topk_ratio:.2f} | Local Neighbors: {local_n}")
        print('='*60)

        # Run cross-validation
        mean, std = run_cv(cfg, device)
        if mean is not None:
            append_single_result(params, mean, std)
            print(f"✅ Completed | Test Acc = {mean['test_accuracy']:.4f} ± {std['test_accuracy']:.4f}")
        else:
            print(f"❌ Training failed, skipped")

    print("\n🎉 Grid search completed!")

if __name__ == '__main__':
    main()