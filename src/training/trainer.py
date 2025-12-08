"""
Training orchestrator for the MIGT-TVDT model.

Handles the complete training loop with:
- Mixed precision training (AMP) for A100 optimization
- Gradient accumulation for effective batch size scaling
- Learning rate scheduling with warmup
- Early stopping on validation CRPS
- Checkpointing with best model tracking
- TensorBoard / WandB logging
- Gradient clipping for stability

Per scientific document Section 6 and engineering specification Section 5.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import time

from .loss_functions import CombinedQuantileLoss
from .scheduler import WarmupCosineScheduler


class EarlyStopping:
    """
    Early stopping handler.
    
    Monitors validation metric and stops training when no improvement
    is observed for `patience` consecutive epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
                Type: int
            min_delta: Minimum change to qualify as improvement
                Type: float
            mode: 'min' for metrics like loss, 'max' for accuracy
                Type: str
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class Trainer:
    """
    Training orchestrator for MIGT-TVDT model.
    
    Manages the complete training lifecycle including:
    - Forward/backward passes with mixed precision
    - Gradient accumulation for larger effective batches
    - Validation evaluation
    - Learning rate scheduling
    - Checkpointing
    - Logging and metrics tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: MIGT-TVDT model instance
                Type: nn.Module
            config: Training configuration dictionary
                Type: Dict[str, Any]
                Required keys in config['training']:
                    batch_size: int (128)
                    gradient_accumulation_steps: int (2)
                    max_epochs: int (100)
                    early_stopping_patience: int (10)
                    mixed_precision: bool (True)
                Required keys in config['optimizer']:
                    lr: float (1e-4)
                    weight_decay: float (0.01)
                Required keys in config['scheduler']:
                    warmup_steps: int (1000)
                    t_0: int (10)
                    t_mult: int (2)
                    eta_min: float (1e-6)
                Required keys in config['regularization']:
                    gradient_clip_norm: float (1.0)
                Required keys in config['quantile_regression']:
                    quantiles: List[float]
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory for checkpoints and logs
            device: Device for training (auto-detect if None)
            logger: Optional WandB or TensorBoard logger
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() 
            else torch.device('cpu')
        )
        self.model = self.model.to(self.device)
        
        # Extract config sections
        train_cfg = config['training']
        opt_cfg = config['optimizer']
        sched_cfg = config['scheduler']
        reg_cfg = config['regularization']
        quant_cfg = config['quantile_regression']
        
        # Initialize loss function
        self.loss_fn = CombinedQuantileLoss(
            quantiles=quant_cfg['quantiles'],
            crossing_weight=quant_cfg.get('crossing_weight', 0.0)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=tuple(opt_cfg.get('betas', [0.9, 0.999]))
        )
        
        # Initialize scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=sched_cfg['warmup_steps'],
            t_0=sched_cfg['t_0'],
            t_mult=sched_cfg['t_mult'],
            eta_min=sched_cfg['eta_min']
        )
        
        # Initialize gradient scaler for mixed precision
        self.use_amp = train_cfg['mixed_precision'] and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Training settings
        self.grad_accum_steps = train_cfg['gradient_accumulation_steps']
        self.max_epochs = train_cfg['max_epochs']
        self.grad_clip_norm = reg_cfg['gradient_clip_norm']
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=train_cfg['early_stopping_patience'],
            mode='min'  # Minimize validation loss
        )
        
        # Checkpointing settings
        self.save_top_k = config.get('checkpointing', {}).get('save_top_k', 3)
        self.best_checkpoints = []  # List of (score, path) tuples
        
        # Logger (WandB or TensorBoard)
        self.logger = logger
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
    def train(self) -> Dict[str, Any]:
        """
        Execute complete training procedure.
        
        Returns:
            Training history dictionary
                Type: Dict[str, Any]
                Keys: 'train_loss', 'val_loss', 'val_metrics', 'learning_rates'
        """
        print(f"Starting training on {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(self.train_loader.dataset):,}")
        print(f"  Validation samples: {len(self.val_loader.dataset):,}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Gradient accumulation: {self.grad_accum_steps}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_loss = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validation epoch
            val_loss, val_metrics = self._validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Log metrics
            if self.logger is not None:
                self._log_metrics(epoch, train_loss, val_loss, val_metrics, current_lr)
            
            # Print progress
            print(
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"PICP80: {val_metrics['picp_80']:.3f} | LR: {current_lr:.2e}"
            )
            
            # Check for best model and save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
                print(f"  New best model saved (val_loss: {val_loss:.6f})")
            else:
                self._save_checkpoint(epoch, val_loss, is_best=False)
            
            # Update scheduler (epoch-level for cosine annealing)
            self.scheduler.step()
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Save final training history
        self._save_history()
        
        return self.history
    
    def _train_epoch(self) -> float:
        """
        Run a single training epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass with mixed precision
            loss = self._train_step(batch)
            total_loss += loss.item() * self.grad_accum_steps  # Undo scaling
            n_batches += 1
            
            # Optimizer step after gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self._optimizer_step()
                
            # Update warmup scheduler (per-batch during warmup)
            self.scheduler.step_batch()
            self.global_step += 1
            
        # Handle remaining gradients if epoch doesn't divide evenly
        if (batch_idx + 1) % self.grad_accum_steps != 0:
            self._optimizer_step()
            
        return total_loss / n_batches
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Single training step with mixed precision.
        
        Args:
            batch: Batch dictionary from dataloader
            
        Returns:
            Scaled loss value (for logging, after accumulation scaling)
        """
        with autocast(enabled=self.use_amp):
            # Prepare temporal info dict
            temporal_info = {
                'bar_in_day': batch['bar_in_day'],
                'day_of_week': batch['day_of_week'],
                'day_of_month': batch['day_of_month'],
                'day_of_year': batch['day_of_year']
            }
            
            # Forward pass
            outputs = self.model(
                features=batch['features'],
                attention_mask=batch['attention_mask'],
                temporal_info=temporal_info
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                predictions=outputs['quantiles'],
                targets=batch['targets']
            )
            loss = loss_dict['total']
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.grad_accum_steps
        
        # Backward pass with gradient scaling
        self.scaler.scale(scaled_loss).backward()
        
        return scaled_loss
    
    def _optimizer_step(self):
        """
        Execute optimizer step with gradient clipping.
        """
        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip_norm
        )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Reset gradients
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def _validate(self) -> tuple:
        """
        Run validation epoch.
        
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        # Collect predictions and targets for metrics
        all_predictions = []
        all_targets = []
        
        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            
            # Prepare temporal info
            temporal_info = {
                'bar_in_day': batch['bar_in_day'],
                'day_of_week': batch['day_of_week'],
                'day_of_month': batch['day_of_month'],
                'day_of_year': batch['day_of_year']
            }
            
            # Forward pass (no AMP needed for validation)
            outputs = self.model(
                features=batch['features'],
                attention_mask=batch['attention_mask'],
                temporal_info=temporal_info
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                predictions=outputs['quantiles'],
                targets=batch['targets']
            )
            total_loss += loss_dict['total'].item()
            n_batches += 1
            
            # Collect for metrics
            all_predictions.append(outputs['quantiles'].cpu())
            all_targets.append(batch['targets'].cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute detailed metrics
        metrics = self.loss_fn.get_metrics(all_predictions, all_targets)
        
        avg_loss = total_loss / n_batches
        return avg_loss, metrics
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """
        Move batch tensors to training device.
        
        Args:
            batch: Batch dictionary from dataloader
            
        Returns:
            Batch with tensors on device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif key == 'norm_stats':
                # Keep norm_stats as list of dicts (not moved to device)
                device_batch[key] = value
            else:
                device_batch[key] = value
        return device_batch
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool
    ):
        """
        Save training checkpoint.
        
        Maintains top-k best checkpoints and always saves latest.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss for this checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            
            # Manage top-k checkpoints
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pt'
            torch.save(checkpoint, epoch_path)
            
            self.best_checkpoints.append((val_loss, epoch_path))
            self.best_checkpoints.sort(key=lambda x: x[0])
            
            # Remove excess checkpoints
            while len(self.best_checkpoints) > self.save_top_k:
                _, old_path = self.best_checkpoints.pop()
                if old_path.exists() and 'best' not in str(old_path):
                    old_path.unlink()
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
        lr: float
    ):
        """
        Log metrics to external logger (WandB or TensorBoard).
        """
        if self.logger is None:
            return
            
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        
        # WandB logging
        if hasattr(self.logger, 'log'):
            self.logger.log(metrics, step=epoch)
        # TensorBoard logging
        elif hasattr(self.logger, 'add_scalar'):
            for key, value in metrics.items():
                self.logger.add_scalar(key, value, epoch)
    
    def _save_history(self):
        """Save training history to JSON file."""
        history_path = self.output_dir / 'training_history.json'
        
        # Convert numpy types for JSON serialization
        history_json = {}
        for key, values in self.history.items():
            if key == 'val_metrics':
                history_json[key] = values  # Already dicts with Python floats
            else:
                history_json[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
            
        print(f"Training history saved to {history_path}")


def create_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    data_module: Any,
    output_dir: Path,
    use_wandb: bool = False
) -> Trainer:
    """
    Factory function to create trainer with optional WandB logging.
    
    Args:
        model: MIGT-TVDT model
        config: Full configuration dictionary
        data_module: NQDataModule with setup() already called
        output_dir: Output directory for checkpoints
        use_wandb: Whether to use WandB for logging
        
    Returns:
        Configured Trainer instance
    """
    logger = None
    
    if use_wandb:
        try:
            import wandb
            project_name = config.get('logging', {}).get('project_name', 'nq-futures-dist')
            wandb.init(project=project_name, config=config)
            logger = wandb
            print(f"WandB logging enabled (project: {project_name})")
        except ImportError:
            print("WandB not installed, logging disabled")
    
    return Trainer(
        model=model,
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        output_dir=output_dir,
        logger=logger
    )
