import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
from utils import accuracy, TotalMeter, count_params, isfloat, continus_mixup_data


class Train:
    def __init__(self, cfg, model, optimizers, lr_schedulers, dataloaders, logger, device):
        """Initialize training class with model, optimizer and configuration"""
        self.config = cfg
        self.logger = logger
        self.model = model
        self.device = device
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.save_learnable_graph = cfg.save_learnable_graph
        self.divergence_weight = cfg.divergence_weight
        self.init_meters()

    def init_meters(self):
        """Initialize loss and accuracy meters"""
        self.train_loss, self.test_loss = TotalMeter(), TotalMeter()
        self.train_accuracy, self.test_accuracy = TotalMeter(), TotalMeter()

    def reset_meters(self):
        """Reset all meters for new epoch"""
        for meter in [self.train_accuracy, self.test_accuracy, self.train_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        """Single training epoch process"""
        self.model.train()
        for time_series, node_feature, label, site, age, sex in self.train_dataloader:
            label = label.float()
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            # Move data to target device
            time_series = time_series.to(self.device)
            node_feature = node_feature.to(self.device)
            label = label.to(self.device)
            site = site.to(self.device)
            age = age.to(self.device)
            sex = sex.to(self.device)

            if self.config.preprocess.continus:
                time_series, node_feature, label, site, age, sex = continus_mixup_data(
                    time_series, node_feature, y=label, site=site, age=age, sex=sex,
                    alpha=1.0, device=self.device
                )

            predict, fc = self.model(time_series, node_feature, site, age, sex)
            # Site-specific loss calculation
            unique_sites = torch.unique(site)
            site_losses = []
            for s in unique_sites:
                mask = (site == s)
                site_losses.append(self.loss_fn(predict[mask], label[mask]))
            invariant_loss = torch.mean(torch.stack(site_losses))
            divergence_loss = torch.mean(F.relu(invariant_loss - torch.stack(site_losses)))
            loss = self.loss_fn(predict, label) + divergence_loss * self.divergence_weight
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        """Single evaluation epoch process"""
        labels = []
        result = []
        self.model.eval()
        with torch.no_grad():
            for time_series, node_feature, label, site, age, sex in dataloader:
                time_series = time_series.to(self.device)
                node_feature = node_feature.to(self.device)
                label = label.to(self.device)
                site = site.to(self.device)
                age = age.to(self.device)
                sex = sex.to(self.device)

                output, fc = self.model(time_series, node_feature, site, age, sex)
                label = label.float()
                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(loss.item(), label.shape[0])
                top1 = accuracy(output, label[:, 1])[0]
                acc_meter.update_with_weight(top1, label.shape[0])
                result += F.softmax(output, dim=1)[:, 1].tolist()
                labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(labels, result, average='binary', zero_division=0)
        report = classification_report(labels, result, output_dict=True, zero_division=0)
        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def generate_save_learnable_matrix(self):
        pass

    def train(self):
        """Main training loop with evaluation and model saving"""
        warnings.filterwarnings("ignore")
        training_process = []
        self.current_step = 0
        best_test_accuracy = 0.0
        best_results = None
        best_model_state = None

        total_training_time = 0.0
        epoch_times = []
        total_infer_time = 0.0
        total_infer_samples = 0
        current_mem_per_epoch = []
        per_epoch_infer_per_sample = []
        max_gpu_memory = 0.0
        use_cuda = self.device.type == 'cuda'

        for epoch in range(self.epochs):
            if epoch > 100:
                break
            epoch_start = time.time()
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])

            if use_cuda:
                current_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                current_mem_per_epoch.append(current_mem)
                max_gpu_memory = max(max_gpu_memory, current_mem)
                torch.cuda.reset_max_memory_allocated()

            # Inference time measurement
            infer_start = time.time()
            train_result = self.test_per_epoch(self.train_dataloader, self.train_loss, self.train_accuracy)
            train_infer_time = time.time() - infer_start

            infer_start = time.time()
            test_result = self.test_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)
            test_infer_time = time.time() - infer_start

            epoch_infer_time = train_infer_time + test_infer_time
            epoch_infer_samples = len(self.train_dataloader.dataset) + len(self.test_dataloader.dataset)
            epoch_infer_per_sample = epoch_infer_time / epoch_infer_samples if epoch_infer_samples > 0 else 0.0
            per_epoch_infer_per_sample.append(epoch_infer_per_sample)
            total_infer_time += epoch_infer_time
            total_infer_samples += epoch_infer_samples

            epoch_total_time = time.time() - epoch_start
            epoch_times.append(epoch_total_time)
            total_training_time += epoch_total_time

            log_items = [
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Acc:{self.train_accuracy.avg: .3f}%',
                f'Test Acc:{self.test_accuracy.avg: .3f}%',
                f'Test Sen:{test_result[-1]:.4f}',
                f'Test Spe:{test_result[-2]:.4f}',
                f'Test F1:{test_result[-4]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Epoch Time:{epoch_total_time:.2f}s',
                f'Infer Time/Sample:{epoch_infer_per_sample:.6f}s'
            ]
            if use_cuda:
                log_items.append(f'GPU Mem:{current_mem:.2f}MB')
            self.logger.info(" | ".join(log_items))

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Epoch Time (s)": epoch_total_time,
                "Infer Time per Sample (s)": epoch_infer_per_sample,
                "GPU Memory (MB)": current_mem if use_cuda else 0.0
            })

            # Save best model based on test accuracy
            current_test_accuracy = self.test_accuracy.avg
            if current_test_accuracy > best_test_accuracy:
                best_model_state = self.model.state_dict()
                best_test_accuracy = current_test_accuracy
                best_results = {
                    "train_accuracy": self.train_accuracy.avg,
                    "test_accuracy": current_test_accuracy,
                    "test_auc": test_result[0],
                    'test_sensitivity': test_result[-1],
                    'test_specificity': test_result[-2],
                    'micro_f1': test_result[-4],
                    'micro_recall': test_result[-5],
                    'micro_precision': test_result[-6],
                }
                self.logger.info(f"Best model saved at epoch {epoch} with Test Accuracy: {best_test_accuracy:.4f}")

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()

        # Calculate performance statistics
        num_completed_epochs = len(epoch_times)
        avg_epoch_time = total_training_time / num_completed_epochs if num_completed_epochs > 0 else 0.0
        std_epoch_time = np.std(epoch_times, ddof=1) if num_completed_epochs >= 2 else np.nan
        avg_infer_per_sample = total_infer_time / total_infer_samples if total_infer_samples > 0 else 0.0
        std_infer_per_sample = np.std(per_epoch_infer_per_sample, ddof=1) if num_completed_epochs >= 2 else np.nan
        avg_gpu_mem = np.mean(current_mem_per_epoch) if use_cuda and num_completed_epochs > 0 else 0.0
        std_gpu_mem = np.std(current_mem_per_epoch, ddof=1) if use_cuda and num_completed_epochs >= 2 else np.nan
        if use_cuda and num_completed_epochs > 0:
            final_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            max_gpu_memory = max(max_gpu_memory, final_mem)

        def format_std(value):
            return f"{value:.2f}" if not np.isnan(value) else "N/A"

        # Final training statistics
        final_stats = [
            "\n" + "=" * 100,
            "TRAINING COMPLETE - PERFORMANCE STATISTICS (Mean ± Std)",
            "=" * 100,
            f"Completed Epochs: {num_completed_epochs}",
            f"Epoch Time: Mean={avg_epoch_time:.2f}s | Std={format_std(std_epoch_time)}s",
            f"Inference Time per Sample: Mean={avg_infer_per_sample:.6f}s ({avg_infer_per_sample * 1000:.3f}ms) | "
            f"Std={format_std(std_infer_per_sample * 1000) if not np.isnan(std_infer_per_sample) else 'N/A'}ms",
            f"GPU Memory: Avg={avg_gpu_mem:.2f}MB | Std={format_std(std_gpu_mem)}MB | Max={max_gpu_memory:.2f}MB"
            if use_cuda else "GPU Memory: Not Used",
            f"Total Inference Samples: {total_infer_samples}",
            f"Total Inference Time: {total_infer_time:.2f}s",
            f"Total Training Time: {total_training_time:.2f}s",
            "=" * 100 + "\n"
        ]
        for stat in final_stats:
            self.logger.info(stat)

        training_summary = {
            "Summary": "Training Statistics (Mean + Std)",
            "Total Training Time (s)": total_training_time,
            "Completed Epochs": num_completed_epochs,
            "Average Epoch Time (s)": avg_epoch_time,
            "Std Epoch Time (s)": std_epoch_time if not np.isnan(std_epoch_time) else 0.0,
            "Total Inference Samples": total_infer_samples,
            "Total Inference Time (s)": total_infer_time,
            "Average Inference Time per Sample (s)": avg_infer_per_sample,
            "Std Inference Time per Sample (s)": std_infer_per_sample if not np.isnan(std_infer_per_sample) else 0.0,
            "Average GPU Memory per Epoch (MB)": avg_gpu_mem,
            "Std GPU Memory per Epoch (MB)": std_gpu_mem if not np.isnan(std_gpu_mem) else 0.0,
            "Max GPU Memory Used (MB)": max_gpu_memory if use_cuda else 0.0,
            "Best Test Accuracy": best_test_accuracy,
            "Best Test Results": best_results
        }
        training_process.append(training_summary)
        return best_results