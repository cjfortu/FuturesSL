"""
Learning rate scheduling for the MIGT-TVDT training pipeline.

Implements warmup + cosine annealing with restarts, as specified in
the scientific document Section 6:
- Linear warmup from 0 to base LR over warmup_steps
- Cosine annealing with restarts (T_0 initial period, T_mult multiplier)
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    
    During warmup: LR increases linearly from 0 to base_lr
    After warmup: Cosine annealing with optional restarts
    
    Per scientific document: Start with warmup to stabilize gradients,
    then anneal to allow fine-grained optimization.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        t_0: int = 10,
        t_mult: int = 2,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Wrapped optimizer
                Type: torch.optim.Optimizer
            warmup_steps: Number of warmup steps (not epochs)
                Type: int
                Default: 1000
            t_0: Initial number of epochs for first cosine cycle
                Type: int
                Default: 10
            t_mult: Factor to increase cycle length after each restart
                Type: int
                Default: 2 (second cycle = 20 epochs, third = 40, etc.)
            eta_min: Minimum learning rate
                Type: float
                Default: 1e-6
            last_epoch: Index of last epoch for resuming
                Type: int
                Default: -1 (start fresh)
        """
        self.warmup_steps = warmup_steps
        self.t_0 = t_0
        self.t_mult = t_mult
        self.eta_min = eta_min
        
        # Track step count for warmup
        self.step_count = 0
        self.warmup_finished = False
        
        # Track epoch for cosine annealing
        self.epoch_in_cycle = 0
        self.cycle = 0
        self.current_t = t_0
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        Compute learning rate for current step/epoch.
        
        Returns:
            List of learning rates for each parameter group
        """
        # During warmup: linear increase
        if not self.warmup_finished:
            warmup_factor = self.step_count / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # After warmup: cosine annealing
        # Cosine schedule within current cycle
        progress = self.epoch_in_cycle / self.current_t
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]
    
    def step_batch(self):
        """
        Call this after each batch during warmup phase.
        
        During warmup, LR changes every batch. After warmup, use step()
        to update LR at the end of each epoch.
        """
        if self.warmup_finished:
            return
            
        self.step_count += 1
        
        if self.step_count >= self.warmup_steps:
            self.warmup_finished = True
            
        # Update LR
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
    def step(self, epoch: Optional[int] = None):
        """
        Call this at the end of each epoch (after warmup completes).
        
        Handles cosine annealing with restarts.
        
        Args:
            epoch: Current epoch (optional, auto-incremented if None)
        """
        if not self.warmup_finished:
            return
            
        if epoch is None:
            self.epoch_in_cycle += 1
        else:
            self.epoch_in_cycle = epoch
            
        # Check for restart
        if self.epoch_in_cycle >= self.current_t:
            self.epoch_in_cycle = 0
            self.cycle += 1
            self.current_t = self.t_0 * (self.t_mult ** self.cycle)
            
        # Update LR
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
    def state_dict(self):
        """
        Return scheduler state for checkpointing.
        """
        return {
            'step_count': self.step_count,
            'warmup_finished': self.warmup_finished,
            'epoch_in_cycle': self.epoch_in_cycle,
            'cycle': self.cycle,
            'current_t': self.current_t,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }
        
    def load_state_dict(self, state_dict):
        """
        Load scheduler state from checkpoint.
        """
        self.step_count = state_dict['step_count']
        self.warmup_finished = state_dict['warmup_finished']
        self.epoch_in_cycle = state_dict['epoch_in_cycle']
        self.cycle = state_dict['cycle']
        self.current_t = state_dict['current_t']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']


class LinearWarmupScheduler(_LRScheduler):
    """
    Simple linear warmup followed by constant LR.
    
    Alternative to cosine annealing for simpler experiments.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        last_epoch: int = -1
    ):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            last_epoch: For resuming
        """
        self.warmup_steps = warmup_steps
        self.step_count = 0
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Compute current learning rate."""
        if self.step_count < self.warmup_steps:
            warmup_factor = self.step_count / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs
    
    def step_batch(self):
        """Update LR after each batch during warmup."""
        self.step_count += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
    def step(self, epoch: Optional[int] = None):
        """No-op after warmup (constant LR)."""
        pass
