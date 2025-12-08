# Development Phase 5 Documentation

## Phase Overview
**Status:** COMPLETE  
**Duration:** Implementation + Testing  
**Objective:** Implement training pipeline with mixed precision, gradient accumulation, scheduling, and checkpointing

## Deliverables

### 1. Training Modules
All modules implemented in `/src/training/`:
- `__init__.py` - Package exports
- `loss_functions.py` - PinballLoss, CombinedQuantileLoss
- `scheduler.py` - WarmupCosineScheduler, LinearWarmupScheduler
- `trainer.py` - Trainer, EarlyStopping, create_trainer

### 2. Configuration
Training configuration: `/configs/training_config.yaml`
- Optimizer: AdamW with lr=1e-4, weight_decay=0.01
- Scheduler: Cosine annealing with 1000-step warmup
- Training: batch_size=128, gradient_accumulation=2
- Early stopping: patience=10 on val_loss

### 3. Testing
Comprehensive test notebook: `/notebooks/05_training.ipynb`
- 8 test suites covering all components
- Loss function validation
- Scheduler behavior verification
- Full training loop integration
- Checkpoint save/load

## Architecture Details

### Loss Functions

**PinballLoss** implements quantile regression loss:
```
L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
```

Key features:
- Asymmetric penalty based on quantile level
- Per-quantile and per-horizon breakdown for analysis
- Gradient-friendly implementation

**CombinedQuantileLoss** wraps PinballLoss with:
- Optional sharpness penalty for interval width
- Comprehensive metrics computation (PICP, coverage, interval width)
- No crossing penalty needed (handled by model architecture)

### Learning Rate Schedule

**WarmupCosineScheduler** implements:
1. Linear warmup: 0 to base_lr over warmup_steps
2. Cosine annealing with restarts (T_0=10, T_mult=2)
3. Minimum learning rate eta_min=1e-6

Schedule visualization:
```
LR ^
   |    /\      /\        /\
   |   /  \    /  \      /  \
   |  /    \  /    \    /    \
   | /      \/      \  /      \
   |/                \/        \__
   +----------------------------> Epoch
   Warmup  Cycle1  Cycle2   Cycle3
```

### Trainer Features

1. **Mixed Precision (AMP)**
   - torch.cuda.amp.autocast for forward pass
   - GradScaler for loss scaling
   - ~50% memory reduction on A100

2. **Gradient Accumulation**
   - Effective batch = batch_size * accumulation_steps
   - Default: 128 * 2 = 256 effective batch

3. **Gradient Clipping**
   - Global norm clipping at 1.0
   - Prevents exploding gradients

4. **Checkpointing**
   - Saves top-k best models by val_loss
   - Always saves latest checkpoint
   - Full state: model, optimizer, scheduler, scaler

5. **Early Stopping**
   - Monitors validation loss
   - Patience=10 epochs
   - min_delta=1e-5

## Training Flow

```
for epoch in range(max_epochs):
    # Training
    for batch in train_loader:
        with autocast():
            outputs = model(batch)
            loss = loss_fn(outputs, targets)
        
        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        scheduler.step_batch()  # Warmup updates
    
    # Validation
    val_loss, metrics = validate()
    
    # Checkpointing
    if val_loss < best_val_loss:
        save_checkpoint(is_best=True)
    
    # Scheduler epoch update
    scheduler.step()
    
    # Early stopping
    if early_stopping(val_loss):
        break
```

## Memory Usage

| Component | Memory |
|-----------|--------|
| Model parameters | ~15 MB |
| Adam optimizer states | ~30 MB |
| Gradient buffers | ~15 MB |
| Activations (B=128) | ~35 GB |
| **Total @ B=128** | **~36 GB** |

With AMP enabled, activations reduce by ~50%, allowing comfortable B=128 on A100 80GB.

## Validation Metrics

Computed per validation epoch:
- `val_loss`: Average pinball loss
- `picp_80`: 80% prediction interval coverage probability
- `coverage_qXX`: Actual coverage at each quantile
- `interval_80_mean`: Mean width of 80% PI
- `interval_50_mean`: Mean width of 50% PI
- `loss_XX`: Per-horizon loss breakdown

## API Reference

### Trainer

```python
trainer = Trainer(
    model=model,
    config=train_config,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir=Path('./outputs'),
    device=torch.device('cuda'),
    logger=wandb  # Optional
)

history = trainer.train()

# Resume from checkpoint
trainer.load_checkpoint(Path('./outputs/checkpoint_latest.pt'))
```

### Loss Functions

```python
loss_fn = CombinedQuantileLoss(
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    crossing_weight=0.0  # Set to 0.1 if crossing penalty desired
)

loss_dict = loss_fn(predictions, targets)
# Returns: {'total': tensor, 'pinball': tensor, 'crossing': tensor}

metrics = loss_fn.get_metrics(predictions, targets)
# Returns: {'picp_80': float, 'coverage_q50': float, ...}
```

### Scheduler

```python
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_steps=1000,
    t_0=10,
    t_mult=2,
    eta_min=1e-6
)

# During warmup (per batch)
scheduler.step_batch()

# After warmup (per epoch)
scheduler.step()
```

## Compatibility

### With Previous Phases
- Phase 3 (Dataset): NQDataModule integrates directly with Trainer
- Phase 4 (Model): MIGT_TVDT output format matches loss function expectations

### Expected Model Output
```python
outputs = model(features, attention_mask, temporal_info)
# outputs['quantiles']: (B, 5, 7) - 5 horizons, 7 quantiles
```

### Expected Target Format
```python
targets: (B, 5)  # One value per horizon
```

## Full Training Setup

```python
# 1. Load configs
with open('configs/model_config.yaml') as f:
    model_config = yaml.safe_load(f)
with open('configs/training_config.yaml') as f:
    train_config = yaml.safe_load(f)

# 2. Initialize data
data_module = NQDataModule(
    data_path='data/processed/nq_features_full.parquet',
    batch_size=train_config['training']['batch_size']
)
data_module.setup()

# 3. Initialize model
model = MIGT_TVDT(model_config['model'])

# 4. Create trainer
trainer = create_trainer(
    model=model,
    config=train_config,
    data_module=data_module,
    output_dir=Path('./outputs'),
    use_wandb=True
)

# 5. Train
history = trainer.train()
```

## Next Steps

1. **Run Full Training**: Execute with complete dataset
2. **Monitor**: Use WandB/TensorBoard for loss curves
3. **Hyperparameter Tuning**: Adjust lr, batch_size, architecture
4. **Phase 6**: Evaluation framework implementation

## References

- AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017)
- Cosine Annealing: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2016)
- Mixed Precision: Micikevicius et al., "Mixed Precision Training" (2017)
- Pinball Loss: Koenker & Bassett, "Regression Quantiles" (1978)

---
**Phase Status:** COMPLETE  
**Ready for:** Phase 6 (Evaluation & Analysis)  
**Last Updated:** 2025-12-08
