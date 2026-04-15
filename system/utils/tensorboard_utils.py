# PFLlib: Personalized Federated Learning Algorithm Library
# TensorBoard Integration for Real-time Training Visualization

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    TensorBoard logger for federated learning experiments.
    Provides real-time visualization of training metrics.

    Usage:
        tensorboard --logdir=./runs
    """

    def __init__(self, args, enabled=True):
        self.enabled = enabled
        if not enabled:
            return

        # Create unique run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.dataset}_{args.algorithm}_{args.model}"

        if hasattr(args, 'privacy') and args.privacy:
            run_name += f"_DP_sigma{args.dp_sigma}"

        run_name += f"_nc{args.num_clients}_gr{args.global_rounds}_{timestamp}"

        log_dir = os.path.join("runs", run_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.run_name = run_name

        # Log hyperparameters
        hparams = {
            'algorithm': args.algorithm,
            'dataset': args.dataset,
            'model': args.model,
            'num_clients': args.num_clients,
            'global_rounds': args.global_rounds,
            'local_epochs': args.local_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.local_learning_rate,
            'join_ratio': args.join_ratio,
        }

        if hasattr(args, 'privacy'):
            hparams['privacy'] = args.privacy
            if args.privacy:
                hparams['dp_sigma'] = args.dp_sigma

        # Log hparams as text
        hparams_text = "\n".join([f"**{k}**: {v}" for k, v in hparams.items()])
        self.writer.add_text("Hyperparameters", hparams_text, 0)

        print(f"\n[TensorBoard] Logging to: {log_dir}")
        print(f"[TensorBoard] Run: tensorboard --logdir=./runs")

    def log_round_metrics(self, round_num, metrics_dict):
        """
        Log metrics for a training round.

        Args:
            round_num: Current global round number
            metrics_dict: Dictionary with metric names and values
        """
        if not self.enabled:
            return

        for name, value in metrics_dict.items():
            if value is not None:
                self.writer.add_scalar(f"Round/{name}", value, round_num)

    def log_train_loss(self, round_num, loss):
        """Log training loss."""
        if not self.enabled:
            return
        self.writer.add_scalar("Training/Loss", loss, round_num)

    def log_test_accuracy(self, round_num, accuracy):
        """Log test accuracy."""
        if not self.enabled:
            return
        self.writer.add_scalar("Evaluation/Test_Accuracy", accuracy, round_num)

    def log_test_auc(self, round_num, auc):
        """Log test AUC."""
        if not self.enabled:
            return
        self.writer.add_scalar("Evaluation/Test_AUC", auc, round_num)

    def log_std_metrics(self, round_num, std_acc, std_auc):
        """Log standard deviation of metrics across clients."""
        if not self.enabled:
            return
        self.writer.add_scalar("Evaluation/Std_Accuracy", std_acc, round_num)
        self.writer.add_scalar("Evaluation/Std_AUC", std_auc, round_num)

    def log_privacy_metrics(self, round_num, epsilon, delta=None):
        """Log differential privacy metrics."""
        if not self.enabled:
            return
        self.writer.add_scalar("Privacy/Epsilon", epsilon, round_num)
        if delta is not None:
            self.writer.add_scalar("Privacy/Delta", delta, round_num)

    def log_time_cost(self, round_num, time_cost):
        """Log time cost per round."""
        if not self.enabled:
            return
        self.writer.add_scalar("Performance/Time_per_Round", time_cost, round_num)

    def log_client_participation(self, round_num, num_selected, num_active):
        """Log client participation metrics."""
        if not self.enabled:
            return
        self.writer.add_scalar("Clients/Selected", num_selected, round_num)
        self.writer.add_scalar("Clients/Active", num_active, round_num)

    def log_psnr(self, round_num, psnr_value):
        """Log PSNR value from DLG attack evaluation."""
        if not self.enabled:
            return
        self.writer.add_scalar("Privacy/PSNR_dB", psnr_value, round_num)

    def log_client_accuracies(self, round_num, client_accuracies):
        """Log histogram of client accuracies."""
        if not self.enabled:
            return
        import torch
        accs_tensor = torch.tensor(client_accuracies)
        self.writer.add_histogram("Clients/Accuracy_Distribution", accs_tensor, round_num)

    def log_model_gradients(self, round_num, model, prefix="Global"):
        """Log model gradient statistics."""
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"{prefix}_Gradients/{name}", param.grad, round_num)

    def log_model_weights(self, round_num, model, prefix="Global"):
        """Log model weight statistics."""
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"{prefix}_Weights/{name}", param.data, round_num)

    def log_learning_rate(self, round_num, lr):
        """Log current learning rate."""
        if not self.enabled:
            return
        self.writer.add_scalar("Training/Learning_Rate", lr, round_num)

    def log_memory_usage(self, round_num, gpu_memory_mb=None, cpu_memory_mb=None):
        """Log memory usage."""
        if not self.enabled:
            return
        if gpu_memory_mb is not None:
            self.writer.add_scalar("Performance/GPU_Memory_MB", gpu_memory_mb, round_num)
        if cpu_memory_mb is not None:
            self.writer.add_scalar("Performance/CPU_Memory_MB", cpu_memory_mb, round_num)

    def log_convergence_metrics(self, round_num, best_acc, rounds_since_best):
        """Log convergence-related metrics."""
        if not self.enabled:
            return
        self.writer.add_scalar("Convergence/Best_Accuracy", best_acc, round_num)
        self.writer.add_scalar("Convergence/Rounds_Since_Best", rounds_since_best, round_num)

    def add_custom_scalars_layout(self):
        """Add custom scalars layout for better dashboard organization."""
        if not self.enabled:
            return

        layout = {
            "Training Overview": {
                "Loss & Accuracy": ["Multiline", ["Training/Loss", "Evaluation/Test_Accuracy"]],
            },
            "Evaluation": {
                "Accuracy": ["Multiline", ["Evaluation/Test_Accuracy", "Evaluation/Std_Accuracy"]],
                "AUC": ["Multiline", ["Evaluation/Test_AUC", "Evaluation/Std_AUC"]],
            },
            "Privacy": {
                "DP Metrics": ["Multiline", ["Privacy/Epsilon", "Privacy/PSNR_dB"]],
            },
            "Performance": {
                "Time & Memory": ["Multiline", ["Performance/Time_per_Round", "Performance/GPU_Memory_MB"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def flush(self):
        """Flush the writer to ensure all data is written."""
        if not self.enabled:
            return
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        if not self.enabled:
            return
        self.writer.close()
        print(f"\n[TensorBoard] Closed logger for run: {self.run_name}")
