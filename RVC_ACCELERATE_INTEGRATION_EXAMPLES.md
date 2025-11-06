# Hugging Face Accelerate Integration - Code Examples

This document provides concrete code examples showing how to integrate Accelerate into the Ultimate RVC project at key points.

---

## 1. Basic Accelerate Setup

### Current Implementation (Manual)

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 318-445)

```python
# Manual device setup
device = torch.device(device_type)
gpus = gpus or {0}
n_gpus = len(gpus)

# Manual DDP init
dist.init_process_group(
    backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
    init_method="env://",
    world_size=n_gpus if device.type == "cuda" else 1,
    rank=rank if device.type == "cuda" else 0,
)

# Manual model placement
net_g = net_g.cuda(device_id) if device.type == "cuda" else net_g.to(device)
net_d = net_d.cuda(device_id) if device.type == "cuda" else net_d.to(device)

# Manual DDP wrapping
if n_gpus > 1 and device.type == "cuda":
    net_g = DDP(net_g, device_ids=[device_id])
    net_d = DDP(net_d, device_ids=[device_id])
```

### Accelerate Equivalent

```python
from accelerate import Accelerator
from accelerate.utils import DistributedType

# Simple initialization
accelerator = Accelerator(
    fp16=use_mixed_precision,
    mixed_precision="fp16" if use_mixed_precision else "no",
    gradient_accumulation_steps=accumulation_steps,
    log_with="tensorboard",
)

# Device detection and setup is automatic
# DDP initialization is automatic
# Model placement is handled by accelerator.prepare()

# Models are prepared automatically:
net_g, net_d, optim_g, optim_d, train_loader = accelerator.prepare(
    net_g, net_d, optim_g, optim_d, train_loader
)

# Check if we're the main process
if accelerator.is_main_process:
    # Initialize logging
    writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))
```

**Benefits**:

- Device management automatic
- DDP/single-GPU/CPU handled transparently
- Mixed precision optional via flag
- 40+ lines reduced to 15 lines

---

## 2. Training Loop Integration

### Current Implementation (Manual)

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 671-731)

```python
for batch_idx, info in data_iterator:
    # Manual device transfer
    if device.type == "cuda" and not cache_data_in_gpu:
        info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
    elif device.type != "cuda":
        info = [tensor.to(device) for tensor in info]

    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info

    # Forward pass
    model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model_output

    # Loss computation
    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

    # Manual optimizer step
    optim_d.zero_grad()
    loss_disc.backward()
    grad_norm_d = commons.grad_norm(net_d.parameters())
    optim_d.step()

    # Generator step
    _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
    loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, _ = generator_loss(y_d_hat_g)
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

    optim_g.zero_grad()
    loss_gen_all.backward()
    grad_norm_g = commons.grad_norm(net_g.parameters())
    optim_g.step()

    global_step += 1
```

### Accelerate Equivalent

```python
for batch_idx, info in data_iterator:
    # Device transfer automatic - accelerator handles it
    with accelerator.accumulate(net_g, net_d):
        # Unpack batch
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info

        # Forward pass
        model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model_output

        # Discriminator step
        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

        # Backward with automatic gradient scaling for mixed precision
        accelerator.backward(loss_disc)

        # Optional gradient clipping
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(net_d.parameters(), max_grad_norm=1.0)

        optim_d.step()
        optim_d.zero_grad()

        # Generator step
        _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        # Backward with automatic gradient scaling
        accelerator.backward(loss_gen_all)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(net_g.parameters(), max_grad_norm=1.0)

        optim_g.step()
        optim_g.zero_grad()

    # Gradient accumulation automatic
    if accelerator.sync_gradients:
        global_step += 1
```

**Benefits**:

- Device transfers automatic
- Gradient scaling automatic (mixed precision)
- Gradient accumulation built-in
- Gradient clipping support added
- Compatible with multi-GPU/distributed training

---

## 3. Checkpoint Save/Load Integration

### Current Implementation

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 934-952)

```python
# Manual checkpoint saving
save_checkpoint(
    net_g,
    optim_g,
    config.train.learning_rate,
    epoch,
    lowest_g_value,
    consecutive_increases_gen,
    os.path.join(experiment_dir, f"G_{idx}.pth"),
)

# Manual checkpoint loading (in utils.py)
checkpoint_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(new_state_dict, strict=False)
if optimizer:
    optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))
```

### Accelerate Equivalent

```python
from accelerate.utils import save_and_load_optimizer, save_accelerator_state, load_accelerator_state

# Saving
accelerator.save_state(output_dir=os.path.join(experiment_dir, f"checkpoint_{epoch}"))

# Or manual save with proper unwrapping
accelerator.save({
    "model_g_state_dict": accelerator.unwrap_model(net_g).state_dict(),
    "model_d_state_dict": accelerator.unwrap_model(net_d).state_dict(),
    "optimizer_g": optim_g.state_dict(),
    "optimizer_d": optim_d.state_dict(),
    "epoch": epoch,
    "best_loss": lowest_g_value,
}, os.path.join(experiment_dir, f"checkpoint_{epoch}.pt"))

# Loading
accelerator.load_state(os.path.join(experiment_dir, "checkpoint_best"))

# Or manual load
checkpoint = torch.load(checkpoint_path, map_location="cpu")
accelerator.unwrap_model(net_g).load_state_dict(checkpoint["model_g_state_dict"])
accelerator.unwrap_model(net_d).load_state_dict(checkpoint["model_d_state_dict"])
optim_g.load_state_dict(checkpoint["optimizer_g"])
optim_d.load_state_dict(checkpoint["optimizer_d"])
```

**Benefits**:

- Device-safe saving/loading automatic
- Handles DDP model unwrapping
- Distributed checkpoint coordination
- Built-in recovery options

---

## 4. Data Loading Integration

### Current Implementation

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 349-358)

```python
train_loader = DataLoader(
    train_dataset,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
    batch_sampler=train_sampler,
    persistent_workers=True,
    prefetch_factor=8,
)
```

### Accelerate Integration

```python
from accelerate import Accelerator

# Create DataLoader as usual
train_loader = DataLoader(
    train_dataset,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
    batch_sampler=train_sampler,
    persistent_workers=True,
    prefetch_factor=8,
)

# Let Accelerate prepare it
train_loader = accelerator.prepare(train_loader)

# Accelerate automatically:
# - Handles distributed sampling for multi-GPU
# - Sets correct num_replicas and rank
# - May adjust batch size if gradient_accumulation_steps > 1
```

**Alternative: Manual Batch Size Adjustment**

```python
# If you want to manually handle accumulation
effective_batch_size = batch_size * gradient_accumulation_steps

if accelerator.distributed_type != DistributedType.NO:
    # Adjust for multiple GPUs
    effective_batch_size //= accelerator.num_processes

train_loader = DataLoader(
    train_dataset,
    batch_size=effective_batch_size,
    # ... other args
)
```

---

## 5. Mixed Precision Training

### Setup

```python
from accelerate import Accelerator

# Option 1: FP16 (faster, good for most modern GPUs)
accelerator = Accelerator(
    mixed_precision="fp16",
)

# Option 2: BF16 (more stable, requires newer GPUs)
accelerator = Accelerator(
    mixed_precision="bf16",
)

# Option 3: No mixed precision (debug/compatibility)
accelerator = Accelerator(
    mixed_precision="no",
)
```

### Training Loop Changes

```python
# With Accelerate, backward pass automatically scales loss
accelerator.backward(loss)

# No need for manual GradScaler:
# ❌ DON'T DO THIS with Accelerate:
# scaler = GradScaler()
# scaler.scale(loss).backward()

# Accelerate handles it internally
```

---

## 6. Device Configuration Integration

### Current Implementation

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/configs/config.py` (lines 26-77)

```python
@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # ...
```

### Accelerate Equivalent

```python
from accelerate import Accelerator
from accelerate.utils import get_device_type

@singleton
class Config:
    def __init__(self, use_accelerate=True):
        if use_accelerate:
            accelerator = Accelerator()
            self.device = accelerator.device
            self.device_type = get_device_type()
            self.is_distributed = accelerator.distributed_type != DistributedType.NO
            self.num_processes = accelerator.num_processes
        else:
            # Fallback to old behavior
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.is_distributed = False
            self.num_processes = 1

        # ... rest of config
```

---

## 7. Logging Integration

### With TensorBoard

```python
from accelerate import Accelerator

accelerator = Accelerator(
    log_with="tensorboard",
    project_dir=experiment_dir,
)

# Initialize logging if main process
if accelerator.is_main_process:
    accelerator.init_trackers(
        project_name="ultimate-rvc",
        config={
            "num_epochs": total_epoch,
            "batch_size": batch_size,
            "learning_rate": config.train.learning_rate,
        }
    )

# Log during training
if accelerator.is_main_process:
    accelerator.log({
        "train/loss_disc": loss_disc.item(),
        "train/loss_gen": loss_gen_all.item(),
        "train/grad_norm_d": grad_norm_d,
        "train/grad_norm_g": grad_norm_g,
    }, step=global_step)

# Finish logging
if accelerator.is_main_process:
    accelerator.end_training()
```

---

## 8. Gradient Accumulation Example

### Setup

```python
from accelerate import Accelerator

# Accumulate gradients over 4 steps
accelerator = Accelerator(
    gradient_accumulation_steps=4,
)

# Prepare all objects
net_g, net_d, optim_g, optim_d, train_loader = accelerator.prepare(
    net_g, net_d, optim_g, optim_d, train_loader
)
```

### Training Loop

```python
for batch_idx, batch in enumerate(train_loader):
    # Accumulate is context manager for convenience
    with accelerator.accumulate(net_g, net_d):
        # Loss computation
        loss = compute_loss(batch)

        # Backward (gradients accumulate)
        accelerator.backward(loss)

        # Step only when accumulated (automatic)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(net_g.parameters(), 1.0)
            optim_g.step()
            optim_g.zero_grad()

# Effective batch size = batch_size * 4
# Useful for GPUs with limited memory
```

---

## 9. Multi-GPU Inference

### Current Implementation (Single GPU Only)

```python
self.net_g = self.net_g.to(self.config.device).float()
self.net_g.eval()

# Single audio at a time
with torch.no_grad():
    output = self.net_g(features)
```

### Accelerate Equivalent (Multi-GPU Ready)

```python
from accelerate import Accelerator

class VoiceConverter:
    def __init__(self, use_accelerate=False):
        if use_accelerate:
            self.accelerator = Accelerator()
            self.net_g = self.accelerator.prepare_model(self.net_g)
        else:
            self.net_g = self.net_g.to(self.config.device)

    def convert_audio_batch(self, audio_list):
        """Convert multiple audio files in parallel."""
        with torch.no_grad():
            outputs = []
            for audio in audio_list:
                # Accelerate handles device placement
                features = self.extract_features(audio)

                if hasattr(self, 'accelerator'):
                    # Automatic batching and device handling
                    output = self.net_g(features)
                else:
                    output = self.net_g(features)

                outputs.append(output)

            return outputs
```

---

## 10. Environment Variables for Configuration

### Add to Project

Create new file: `/home/june/new/ultimate-rvc/src/ultimate_rvc/accelerate_config.py`

```python
"""Configuration for Accelerate integration."""
import os
from enum import Enum

class AcceleratorType(str, Enum):
    """Hardware accelerator types."""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    AUTO = "auto"

class MixedPrecisionType(str, Enum):
    """Mixed precision modes."""
    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"
    AUTO = "auto"

def get_accelerator_config():
    """Get accelerator configuration from environment variables."""
    return {
        "accelerator_type": os.getenv(
            "URVC_ACCELERATOR",
            AcceleratorType.AUTO.value
        ),
        "mixed_precision": os.getenv(
            "URVC_MIXED_PRECISION",
            MixedPrecisionType.NO.value
        ),
        "gradient_accumulation_steps": int(
            os.getenv("URVC_GRAD_ACCUMULATION", "1")
        ),
        "max_grad_norm": float(
            os.getenv("URVC_MAX_GRAD_NORM", "1.0")
        ),
        "use_fp16_inference": os.getenv(
            "URVC_FP16_INFERENCE",
            "false"
        ).lower() == "true",
    }

def create_accelerator(**kwargs):
    """Create Accelerator with configuration."""
    from accelerate import Accelerator

    config = get_accelerator_config()
    config.update(kwargs)

    return Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )
```

### Usage Example

```bash
# Enable mixed precision training
export URVC_MIXED_PRECISION=fp16
export URVC_GRAD_ACCUMULATION=2

# Run training
./urvc train main --model-name my_model --num-epochs 500
```

---

## 11. Backward Compatibility Helper

If you need gradual migration:

```python
"""Backward compatibility layer for Accelerate integration."""
from contextlib import contextmanager
import os

USE_ACCELERATE = os.getenv("URVC_USE_ACCELERATE", "false").lower() == "true"

if USE_ACCELERATE:
    from accelerate import Accelerator
    _accelerator_instance = None

    def get_accelerator():
        global _accelerator_instance
        if _accelerator_instance is None:
            _accelerator_instance = Accelerator()
        return _accelerator_instance
else:
    # Dummy class for non-Accelerate mode
    class DummyAccelerator:
        @contextmanager
        def accumulate(self, *models):
            yield

        def backward(self, loss):
            loss.backward()

        def clip_grad_norm_(self, params, max_norm):
            pass

        @property
        def is_main_process(self):
            return True

        @property
        def sync_gradients(self):
            return True

    def get_accelerator():
        return DummyAccelerator()
```

---

## 12. Testing Mixed Precision

### Small Test Script

```python
"""Test mixed precision training."""
import torch
from accelerate import Accelerator

def test_mixed_precision():
    accelerator = Accelerator(mixed_precision="fp16")

    # Simple model
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())

    model, optimizer = accelerator.prepare(model, optimizer)

    # Simple forward/backward
    x = torch.randn(2, 10)
    y = torch.randn(2, 5)

    # With mixed precision
    loss = ((model(x) - y) ** 2).mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    print(f"Loss: {loss.item():.4f}")
    print("Mixed precision training works!")

if __name__ == "__main__":
    test_mixed_precision()
```

---

## 13. Memory Monitoring with Accelerate

```python
"""Monitor memory usage during training."""
import torch
from accelerate import Accelerator

def log_memory_stats(accelerator, label=""):
    """Log GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{label} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# During training
accelerator.print(f"Rank {accelerator.process_index}: Starting epoch")
log_memory_stats(accelerator, "Before epoch")

for batch in train_loader:
    # ... training step ...

    if accelerator.sync_gradients:
        log_memory_stats(accelerator, f"Step {global_step}")
```

---

## 14. Distributed Evaluation

```python
"""Evaluation on distributed setup."""
from accelerate import Accelerator
from accelerate.utils import gather_object

def evaluate_distributed(accelerator, model, eval_loader):
    """Evaluate model on distributed setup."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(batch)
            predictions = outputs.argmax(dim=-1)

            # Gather from all processes
            all_predictions = gather_object(
                predictions.cpu(),
                accelerator=accelerator
            )
            all_labels = gather_object(
                batch["labels"].cpu(),
                accelerator=accelerator
            )

    if accelerator.is_main_process:
        # Compute metrics on main process only
        accuracy = (all_predictions == all_labels).float().mean()
        print(f"Accuracy: {accuracy:.4f}")
```

---

## Summary Table

| Feature | Current | Accelerate | Complexity |
|---------|---------|-----------|-----------|
| Device Management | Manual | Automatic | Low |
| Mixed Precision | None | Built-in | Low |
| Gradient Accumulation | None | Built-in | Low |
| DDP Setup | Manual | Automatic | Low |
| Checkpointing | Manual | Built-in | Low |
| Gradient Clipping | Manual | Built-in | Low |
| Logging | Manual TensorBoard | Integrated | Low |
| Multi-GPU Inference | Not supported | Supported | Medium |

---

## Next Steps

1. **Phase 1**: Create `accelerate_config.py` module
2. **Phase 2**: Add Accelerate to training loop
3. **Phase 3**: Add environment variable support
4. **Phase 4**: Test with multi-GPU setup
5. **Phase 5**: Optimize inference with Accelerate

Each phase can be done independently and backwards-compatible.
