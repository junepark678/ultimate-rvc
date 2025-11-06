# Device Management Quick Reference

## File Locations Map

```
ultimate-rvc/
├── src/ultimate_rvc/
│   ├── rvc/
│   │   ├── train/
│   │   │   ├── train.py                    [Training Loop - 996 lines]
│   │   │   │   ├── main()                  [Lines 125-269: Entry point]
│   │   │   │   ├── run()                   [Lines 272-606: Worker process]
│   │   │   │   │   ├── Device setup        [Lines 318-324: DDP init]
│   │   │   │   │   ├── Model creation      [Lines 392-413]
│   │   │   │   │   └── Device placement    [Lines 415-420]
│   │   │   │   └── train_and_evaluate()    [Lines 609-995: Main loop]
│   │   │   │       ├── Batch transfer      [Lines 671-676]
│   │   │   │       ├── Forward/backward    [Lines 715-731]
│   │   │   │       └── Memory mgmt         [Lines 782, 987]
│   │   │   ├── data_utils.py               [Dataset/Sampler]
│   │   │   ├── utils.py                    [Checkpointing - 150+ lines]
│   │   │   ├── losses.py                   [Loss functions]
│   │   │   └── mel_processing.py           [Mel-spectrogram loss]
│   │   ├── infer/
│   │   │   ├── infer.py                    [VoiceConverter - 526 lines]
│   │   │   │   ├── load_model()            [Lines 482-494: Model loading]
│   │   │   │   ├── setup_network()         [Lines 496-517: Device placement]
│   │   │   │   ├── load_hubert()           [Lines 79-90: Embedder loading]
│   │   │   │   └── convert_audio()         [Lines 217-376: Main inference]
│   │   │   ├── pipeline.py                 [Processing pipeline - 715 lines]
│   │   │   │   ├── __init__()              [Lines 144-232: F0 model init]
│   │   │   │   ├── get_f0_crepe()          [Lines 234-282: F0 extraction]
│   │   │   │   ├── get_f0_hybrid()         [Lines 284-355: F0 methods]
│   │   │   │   ├── voice_conversion()      [Lines 423-514: Core inference]
│   │   │   │   └── pipeline()              [Lines 528-715: Main pipeline]
│   │   │   └── typing_extra.py
│   │   ├── configs/
│   │   │   ├── config.py                   [Device detection - 114 lines]
│   │   │   │   ├── Config.__init__()       [Lines 28-36: Device setup]
│   │   │   │   ├── device_config()         [Lines 55-69: Device tuning]
│   │   │   │   └── set_cuda_config()       [Lines 71-77: CUDA setup]
│   │   │   └── *.json                      [Sample rate configs]
│   │   ├── lib/
│   │   │   ├── zluda.py                    [AMD GPU support - 86 lines]
│   │   │   ├── algorithm/
│   │   │   │   ├── synthesizers.py         [Generator (Synthesizer)]
│   │   │   │   └── discriminators.py       [Discriminator]
│   │   │   └── predictors/
│   │   │       ├── FCPE.py
│   │   │       └── RMVPE.py
│   │   └── common.py                       [RVC constants]
│   └── core/train/
│       ├── train.py                        [CLI entry - calls rvc.train.train.main()]
│       │   └── run_training()              [Lines 145-315+: Top-level orchestration]
│       │       └── validate_devices()      [Line 297: Device validation]
│       └── common.py                       [Device utilities - 105 lines]
│           ├── validate_devices()          [Lines 47-104: Device selection]
│           └── get_gpu_info()              [Lines 19-44: GPU detection]
├── cli/train/
│   └── main.py                             [CLI commands]
│       └── run_training()                  [Lines 145-315+: Calls core.train.train]
│
└── [Analysis Documents]
    ├── RVC_DEVICE_MANAGEMENT_ANALYSIS.md        [Main analysis - 3000+ lines]
    ├── RVC_IMPLEMENTATION_DETAILS.md             [Code reference - 1500+ lines]
    ├── RVC_ACCELERATE_INTEGRATION_EXAMPLES.md    [Examples - 1000+ lines]
    ├── ANALYSIS_SUMMARY.md                       [Summary]
    └── DEVICE_MANAGEMENT_QUICK_REFERENCE.md      [This file]
```

## Device Management Data Flow

### Training Initialization

```
CLI Entry (cli/train/main.py)
  └─ core/train/train.py:run_training()
     └─ validate_devices(hardware_acceleration, gpu_ids)
        └─ core/train/common.py:validate_devices()
           └─ core/train/common.py:get_gpu_info()  [Detects GPUs]
     └─ rvc/train/train.py:main(device_type, gpus)
        └─ Environment setup (MASTER_ADDR, MASTER_PORT)
        └─ For each GPU:
           └─ mp.Process() → run(rank, device_id, ...)
              ├─ dist.init_process_group()           [DDP init]
              ├─ Models: Synthesizer, MultiPeriodDiscriminator
              ├─ Device placement (.cuda(device_id) or .to(device))
              ├─ DDP wrapping (if n_gpus > 1)
              └─ train_and_evaluate() [Training loop]
```

### Inference Initialization

```
VoiceConverter.__init__()
  └─ Config() [Singleton]
     └─ self.device = "cuda:0" or "cpu"

VoiceConverter.convert_audio()
  ├─ load_hubert()
  │  └─ model.to(self.config.device)
  ├─ get_vc() → load_model() → setup_network()
  │  └─ net_g.to(self.config.device)
  └─ vc.pipeline()
     ├─ get_f0() with device transfer
     └─ voice_conversion() [Per chunk]
        └─ feats.to(self.device)
```

## Device Transfer Points

### Training Loop

| Location | Type | Device | Non-blocking |
|----------|------|--------|--------------|
| Line 662 | Cache | cuda | Yes |
| Line 673 | Batch | cuda | Yes |
| Line 675 | Batch | generic | No |
| Line 562-566 | Reference | cuda | Yes |
| Line 569-574 | Reference | generic | No |

### Inference Pipeline

| Location | Type | Device | Method |
|----------|------|--------|--------|
| line 89 | Model | config.device | .to() |
| Line 516 | Model | config.device | .to() |
| Line 257 | Tensor | self.device | .to() |
| Line 460 | Tensor | self.device | .to() |
| Line 523 | Tensor | self.device | .to() |
| Line 629 | Tensor | self.device | tensor() |

## Key Configuration Points

### Training Configuration

```python
# From core/train/train.py
run_training(
    hardware_acceleration: DeviceType,      # AUTOMATIC, GPU, CPU
    gpu_ids: set[int] | None = None,       # GPU indices [0, 1, 2]
    preload_dataset: bool = False,          # Cache to GPU
    reduce_memory_usage: bool = False,      # Activation checkpointing
)

# Passed to rvc/train/train.py:main()
main(
    device_type: str,       # "cuda" or "cpu"
    gpus: set[int] | None,  # GPU IDs like {0, 1, 2}
    cache_data_in_gpu: bool,
    checkpointing: bool,
)
```

### Inference Configuration

```python
# Config singleton in rvc/configs/config.py
Config.device = "cuda:0" or "cpu"      # Hard-coded
Config.gpu_mem                          # Memory in GB
Config.x_pad, x_query, x_center, x_max  # Inference tuning

# Used in infer.py and pipeline.py
Pipeline.device = config.device         # From Config
F0Model.device = self.device            # CREPE, RMVPE, FCPE
```

## Memory Management Strategies

### Current Approach

```
Training Loop:
  1. Optional: Cache all batches to GPU on epoch start
  2. Per batch: Transfer tensors if not cached
  3. Per epoch: Call torch.cuda.empty_cache()
  4. Activation checkpointing: Binary flag in model init

Inference:
  1. Load models to CPU initially
  2. Transfer to device for inference
  3. Call torch.cuda.empty_cache() after conversion
```

### Missing Optimizations

- ❌ No mixed precision (FP16/BF16)
- ❌ No gradient accumulation
- ❌ No gradient scaling
- ❌ No selective checkpointing
- ❌ No automatic batch sizing

## Multi-GPU Distribution

### Current Setup

```python
# rvc/train/train.py:318-324
dist.init_process_group(
    backend="nccl" if cuda else "gloo",
    world_size=n_gpus,
    rank=rank,
    init_method="env://",
)

# DDP Wrapping
net_g = DDP(net_g, device_ids=[device_id])
net_d = DDP(net_d, device_ids=[device_id])
```

### Process Layout

```
Process 0 (GPU 0):           Process 1 (GPU 1):           ...
├─ Model (DDP)               ├─ Model (DDP)
├─ Optimizer                 ├─ Optimizer
├─ Data (Rank 0 subset)      ├─ Data (Rank 1 subset)
├─ Logging (rank==0 only)    └─ Silent
└─ Checkpointing (rank==0)
```

### Synchronization Points

- Forward pass: Gradients computed locally
- Backward pass: Gradients synced via DDP
- Step: All processes update independently

## Missing Environment Variable Support

| Variable | Current | Needed | Use |
|----------|---------|--------|-----|
| URVC_ACCELERATOR | ❌ | ✅ | Select device (auto/cuda/rocm/cpu) |
| URVC_MIXED_PRECISION | ❌ | ✅ | Enable fp16/bf16 training |
| URVC_GRAD_ACCUMULATION | ❌ | ✅ | Gradient accumulation steps |
| URVC_MAX_GRAD_NORM | ❌ | ✅ | Gradient clipping threshold |
| URVC_FP16_INFERENCE | ❌ | ✅ | Mixed precision inference |
| MASTER_ADDR | ✅ | ✓ | localhost (hardcoded) |
| MASTER_PORT | ✅ | ✓ | Random 20000-55555 |

## Accelerate Integration Points

### Priority 1: Training Loop (High Impact)

```
File: src/ultimate_rvc/rvc/train/train.py
Functions: main(), run(), train_and_evaluate()
Changes:
  - Replace dist.init_process_group() with accelerator setup
  - Replace manual device transfers with accelerator.prepare()
  - Replace .backward() with accelerator.backward()
  - Replace .cuda(device_id) with accelerator placement
Benefits: 30% code reduction, 2x speedup potential
```

### Priority 2: Config System (Medium Impact)

```
File: src/ultimate_rvc/rvc/configs/config.py
Changes:
  - Use Accelerator device detection
  - Add environment variable support
  - Centralize device management
Benefits: Cleaner code, more flexible
```

### Priority 3: Inference Optimization (Low-Medium Impact)

```
File: src/ultimate_rvc/rvc/infer/infer.py
Changes:
  - Optional mixed precision
  - Future multi-GPU support
Benefits: Optional optimization
```

## Backward Compatibility Requirements

- ✅ Must support CPU-only training
- ✅ Must support single-GPU (no DDP)
- ✅ Must support multi-GPU DDP
- ✅ Must support Windows (gloo backend)
- ✅ Must support checkpoint resume
- ❌ No need for PyTorch < 2.0

## Performance Targets

### Memory Reduction

- FP16: 50% reduction
- BF16: 50% reduction (more stable)
- Activation checkpointing: 30-40%

### Speed Improvement

- FP16: 1.5-2x faster on RTX/A100
- Gradient accumulation: Effective batch scaling
- Combined: 2-3x potential improvement

## Testing Checklist

- [ ] Single GPU training
- [ ] Multi-GPU training (DDP)
- [ ] CPU-only training
- [ ] Windows compatibility
- [ ] Checkpoint resume
- [ ] Mixed precision (FP16)
- [ ] Gradient accumulation
- [ ] Inference performance
- [ ] Memory usage validation

## Important Code Patterns

### Device-Specific Tensor Placement

```python
# Current pattern (inconsistent)
if device.type == "cuda":
    tensor = tensor.cuda(device_id, non_blocking=True)
else:
    tensor = tensor.to(device)

# Accelerate pattern (unified)
tensor = accelerator.prepare(tensor)
```

### Model State Dict Access

```python
# Current pattern (handles DDP)
state_dict = (
    model.module.state_dict()
    if hasattr(model, "module")
    else model.state_dict()
)

# Accelerate pattern
state_dict = accelerator.unwrap_model(model).state_dict()
```

### Logging on Main Process Only

```python
# Current pattern
if rank == 0:
    writer.log(...)

# Accelerate pattern
if accelerator.is_main_process:
    accelerator.log(...)
```

## Recommended Reading Order

1. **ANALYSIS_SUMMARY.md** (This summary, 5 min)
2. **RVC_DEVICE_MANAGEMENT_ANALYSIS.md** (Detailed analysis, 30 min)
3. **RVC_IMPLEMENTATION_DETAILS.md** (Code reference, browse as needed)
4. **RVC_ACCELERATE_INTEGRATION_EXAMPLES.md** (Implementation, 20 min)
5. **Code Review** (Specific files mentioned above)

## Quick Stats

| Aspect | Value |
|--------|-------|
| Total RVC files analyzed | 15+ |
| Total lines analyzed | 5000+ |
| Training loop size | 996 lines |
| Inference pipeline size | 715 lines |
| Device transfer points | 25+ |
| Hard-coded device checks | 15+ |
| GPU memory configs | 2 |
| Multi-GPU support | Yes (DDP) |
| Mixed precision support | No |
| Gradient accumulation | No |

## Contact & Questions

For questions about:

- **Architecture**: See RVC_DEVICE_MANAGEMENT_ANALYSIS.md
- **Code locations**: See RVC_IMPLEMENTATION_DETAILS.md
- **Integration**: See RVC_ACCELERATE_INTEGRATION_EXAMPLES.md
- **Planning**: See ANALYSIS_SUMMARY.md

---

**Last Updated**: November 2024
**Analysis Scope**: Device management, training, inference
**Complexity**: Comprehensive
**Recommendations**: Phase-based Accelerate integration
