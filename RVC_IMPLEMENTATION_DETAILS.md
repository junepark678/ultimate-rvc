# Ultimate RVC Implementation Details - Code Reference Guide

## Quick Navigation Guide

This document provides quick references to key implementation details and code snippets.

---

## Training System Entry Points

### 1. Main Training Entry Point

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py`
**Function**: `main()` (lines 125-269)
**Entry from CLI**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/core/train/train.py` line 295

**Signature**:

```python
def main(
    model_name: str,
    sample_rate: int,
    vocoder: str,
    total_epoch: int,
    batch_size: int,
    save_every_epoch: int,
    save_only_latest: bool,
    save_every_weights: bool,
    pretrain_g: str,
    pretrain_d: str,
    overtraining_detector: bool,
    overtraining_threshold: int,
    cleanup: bool,
    cache_data_in_gpu: bool,
    checkpointing: bool,
    device_type: str,    # "cuda" or "cpu"
    gpus: set[int] | None,  # GPU IDs like {0, 1, 2}
) -> None:
```

### 2. Multi-Process Training Launcher

**Function**: `main()` internal `start()` function (lines 193-255)

```python
def start() -> None:
    """Start the training process with multi-GPU support or CPU."""
    children = []
    for rank, device_id in enumerate(gpus):
        subproc = mp.Process(
            target=run,  # Worker function
            args=(
                rank,
                n_gpus,
                experiment_dir,
                pretrain_g,
                pretrain_d,
                custom_total_epoch,
                custom_save_every_weights,
                config,
                device,
                device_id,
                # ... more args
            ),
        )
        children.append(subproc)
        subproc.start()
```

**Key Points**:

- Uses `torch.multiprocessing` with `spawn` method
- One process per GPU
- Rank 0 handles logging and checkpointing
- MASTER_ADDR = "localhost", MASTER_PORT = random (20000-55555)

### 3. Per-GPU Worker Process

**Function**: `run()` (lines 272-606)

**Initialization Steps**:

1. Initialize distributed process group (lines 318-324)
2. Set random seed (line 326)
3. Set CUDA device if GPU (lines 328-329)
4. Create dataset and loader (lines 331-358)
5. Build models (lines 392-413)
6. Move models to device (lines 415-420)
7. Setup optimizers (lines 422-438)
8. Wrap with DDP if multi-GPU (lines 442-445)

### 4. Training Loop Worker

**Function**: `train_and_evaluate()` (lines 609-995)

**Each Epoch**:

1. Set sampler epoch for shuffling (line 652)
2. Set models to train mode (lines 654-655)
3. Optionally cache data to GPU (lines 658-667)
4. Loop through batches (lines 671-776)
5. Scheduler step (lines 778-779)
6. Logging and checkpointing (lines 784-934)
7. Early stopping check (lines 972-995)

---

## Detailed Device Management Points

### A. Device Type Selection

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/core/train/common.py` (lines 47-104)

```python
def validate_devices(
    device_type: DeviceType,
    device_ids: set[int] | None = None,
) -> tuple[Literal["cuda", "cpu"], set[int] | None]:
    """Validate and select devices."""
    match device_type:
        case DeviceType.AUTOMATIC:
            gpu_info = get_gpu_info()
            if gpu_info:
                return "cuda", {gpu_info[0][1]}  # First GPU
            return "cpu", None
        case DeviceType.GPU:
            # Validate all device_ids are available
            return "cuda", set(validated_devices)
        case DeviceType.CPU:
            return "cpu", None
```

**GPU Detection**:

```python
def get_gpu_info() -> list[tuple[str, int]]:
    """Returns list of (gpu_name, gpu_index) tuples."""
    ngpu = torch.cuda.device_count()
    gpu_infos: list[tuple[str, int]] = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            gpu_infos.append((f"{gpu_name} ({mem} GB)", i))
    return gpu_infos
```

### B. Runtime Device Configuration

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/configs/config.py` (lines 26-77)

```python
@singleton
class Config:
    def __init__(self):
        # Hard-coded to cuda:0 or cpu
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )

    def device_config(self) -> tuple:
        """Returns (x_pad, x_query, x_center, x_max)."""
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        elif self.has_mps():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Memory-based tuning
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # 4GB GPU config
            return (1, 5, 30, 32)
        else:
            # 6GB GPU config (default)
            return (1, 6, 38, 41)

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024**3)
```

### C. Distributed Training Setup

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 318-324)

```python
dist.init_process_group(
    backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
    init_method="env://",
    world_size=n_gpus if device.type == "cuda" else 1,
    rank=rank if device.type == "cuda" else 0,
)
```

**Backend Selection**:

- Windows or CPU: `gloo` (slower but compatible)
- CUDA: `nccl` (optimal for NVIDIA GPUs)

### D. Model Device Placement

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 415-420)

```python
if device.type == "cuda":
    net_g = net_g.cuda(device_id)
    net_d = net_d.cuda(device_id)
else:
    net_g = net_g.to(device)
    net_d = net_d.to(device)
```

**Pattern**:

- CUDA: Use `.cuda(device_id)` for explicit device placement
- CPU/MPS: Use `.to(device)` for generic placement

### E. Batch Data Transfer

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 671-676)

```python
for batch_idx, info in data_iterator:
    if device.type == "cuda" and not cache_data_in_gpu:
        # CUDA: Use cuda(device_id) with non_blocking
        info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
    elif device.type != "cuda":
        # Non-CUDA: Use .to(device)
        info = [tensor.to(device) for tensor in info]
    # else: already on GPU if cache_data_in_gpu=True
```

**Three Paths**:

1. **Cache to GPU**: Data loaded once at epoch start
2. **Stream to GPU**: Data transferred batch-by-batch
3. **CPU only**: All on CPU (slow, for testing)

### F. DDP Wrapping

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 442-445)

```python
if n_gpus > 1 and device.type == "cuda":
    net_g = DDP(net_g, device_ids=[device_id])
    net_d = DDP(net_d, device_ids=[device_id])
```

**Key Points**:

- Only wraps if `n_gpus > 1` AND CUDA
- Single-GPU or CPU: No wrapping
- Each process has one DDP wrapper

### G. Gradient Computation

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (lines 715-731)

```python
# Discriminator update
y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

global_disc_loss[epoch - 1] += loss_disc.item()
optim_d.zero_grad()
loss_disc.backward()  # Full precision backward
grad_norm_d = commons.grad_norm(net_d.parameters())
optim_d.step()

# Generator update
_, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
loss_fm = feature_loss(fmap_r, fmap_g)
loss_gen, _ = generator_loss(y_d_hat_g)
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

global_gen_loss[epoch - 1] += loss_gen_all.item()
optim_g.zero_grad()
loss_gen_all.backward()  # Full precision backward
grad_norm_g = commons.grad_norm(net_g.parameters())
optim_g.step()
```

**Missing**:

- No loss scaling for mixed precision
- No gradient clipping
- No synchronization helpers

### H. Memory Management

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py` (multiple)

**1. Optional Data Caching (lines 658-667)**:

```python
if device.type == "cuda" and cache_data_in_gpu:
    if cache == []:
        for batch_idx, info in enumerate(train_loader):
            info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            cache.append((batch_idx, info))
    shuffle(cache)
    data_iterator = cache
else:
    data_iterator = enumerate(train_loader)
```

**2. Cache Clearing (lines 782, 987)**:

```python
with torch.no_grad():
    torch.cuda.empty_cache()
```

**3. Gradient Checkpointing (line 406)**:

```python
net_g = Synthesizer(
    # ...
    checkpointing=checkpointing,  # Binary flag
)
```

---

## Inference System Architecture

### Inference Entry Point

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/infer/infer.py`
**Class**: `VoiceConverter` (lines 56-526)
**Main Method**: `convert_audio()` (lines 217-376)

### Model Loading Sequence

**Step 1: Load Model Weights**

```python
def load_model(self, weight_root):
    """Load from disk to CPU."""
    self.cpt = (
        torch.load(weight_root, map_location="cpu", weights_only=False)
        if os.path.isfile(weight_root)
        else None
    )
```

**Step 2: Setup Network**

```python
def setup_network(self):
    """Create model and move to device."""
    if self.cpt is not None:
        self.tgt_sr = self.cpt["config"][-1]
        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=self.text_enc_hidden_dim,
            vocoder=self.vocoder,
        )
        del self.net_g.enc_q  # Remove unnecessary encoder
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.config.device).float()  # Device transfer
        self.net_g.eval()
```

**Step 3: Load Embedder**

```python
def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
    """Load HuBERT for feature extraction."""
    self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
    self.hubert_model = self.hubert_model.to(self.config.device).float()
    self.hubert_model.eval()
```

### Inference Pipeline

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/infer/pipeline.py`
**Class**: `Pipeline` (lines 138-715)

**Device**: Taken from `Config.device` (singleton)

**Flow** (line 528-715):

```python
def pipeline(self, model, net_g, sid, audio, pitch, f0_methods, ...):
    """Main inference pipeline."""

    # 1. Load FAISS index if available
    if file_index != "" and os.path.exists(file_index) and index_rate > 0:
        index = faiss.read_index(file_index)
        big_npy = index.reconstruct_n(0, index.ntotal)

    # 2. Preprocess audio
    audio = signal.filtfilt(bh, ah, audio)  # High-pass filter
    audio_pad = np.pad(audio, (...), mode="reflect")

    # 3. F0 extraction (device transfer here)
    if pitch_guidance:
        pitch, pitchf = self.get_f0(
            audio_pad, p_len, pitch, f0_methods, ...
        )
        pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
        pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

    # 4. Voice conversion on chunks
    for t in opt_ts:
        audio_opt.append(
            self.voice_conversion(
                model, net_g, sid,
                audio_pad[s : t + self.t_pad2 + self.window],
                pitch[:, ...],
                pitchf[:, ...],
                index, big_npy, ...
            )
        )

    # 5. Post-processing
    if volume_envelope != 1:
        audio_opt = AudioProcessor.change_rms(...)

    return audio_opt
```

### F0 Extraction Methods

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/infer/pipeline.py` (lines 234-356)

**Available Methods**:

1. **CREPE** (lines 234-282):

```python
def get_f0_crepe(self, x, f0_min, f0_max, p_len, hop_length, model="full"):
    x = x.astype(np.float32)
    x /= np.quantile(np.abs(x), 0.999)
    audio = torch.from_numpy(x).to(self.device, copy=True)
    audio = torch.unsqueeze(audio, dim=0)

    pitch: Tensor = torchcrepe.predict(
        audio,
        self.sample_rate,
        hop_length,
        f0_min, f0_max,
        model,
        batch_size=hop_length * 2,
        device=self.device,
        pad=True,
    )
    # Interpolation...
    return f0
```

2. **RMVPE** (pre-instantiated, line 229):

```python
self.model_rmvpe = RMVPE0Predictor(
    os.path.join(str(RVC_MODELS_DIR), "predictors", "rmvpe.pt"),
    device=self.device,
)
```

3. **FCPE** (lazy-loaded, lines 334-346):

```python
self.model_fcpe = FCPEF0Predictor(
    os.path.join(str(RVC_MODELS_DIR), "predictors", "fcpe.pt"),
    f0_min=int(f0_min),
    f0_max=int(f0_max),
    dtype=torch.float32,
    device=self.device,
    sample_rate=self.sample_rate,
    threshold=0.03,
)
f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
del self.model_fcpe
gc.collect()
```

### Voice Conversion Core

**Location**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/infer/pipeline.py` (lines 423-514)

```python
def voice_conversion(
    self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, ...
):
    """Single voice conversion segment."""
    with torch.no_grad():
        # Feature extraction
        feats = torch.from_numpy(audio0).float()
        feats = feats.mean(-1) if feats.dim() == 2 else feats
        feats = feats.view(1, -1).to(self.device)

        # Get embeddings
        feats = model(feats)["last_hidden_state"]
        feats = (
            model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
        )

        # Speaker embedding retrieval (optional)
        if index:
            feats = self._retrieve_speaker_embeddings(
                feats, index, big_npy, index_rate,
            )

        # Upsample features
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # Inference
        audio1 = (
            (net_g.infer(feats.float(), p_len, pitch, pitchf.float(), sid)[0][0, 0])
            .data.cpu()
            .float()
            .numpy()
        )

        # Cleanup
        del feats, feats0, p_len
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return audio1
```

---

## Model Architectures

### Generator (Synthesizer)

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/lib/algorithm/synthesizers.py`
**Class**: `Synthesizer` (lines 18-250+)

**Components**:

- Text/phoneme encoder
- Posterior encoder (for training)
- VAE decoder
- Vocoder (HiFi-GAN or RefineGAN)

**Key Methods**:

```python
def __init__(
    self,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel,
    upsample_kernel_sizes,
    n_speakers,
    gin_channels,
    use_f0,
    sr,
    vocoder,
    checkpointing=False,
    randomized=True,
):
    # Lines 47-173: Initialization

def forward(
    self,
    phone,
    phone_lengths,
    pitch,
    pitchf,
    spec,
    spec_lengths,
    sid,
):
    # Lines 175-210: Training forward pass

def infer(
    self,
    phone,
    phone_lengths,
    pitch,
    pitchf,
    sid,
):
    # Lines 213-250+: Inference forward pass (different from training)
```

### Discriminator

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/lib/algorithm/discriminators.py`
**Class**: `MultiPeriodDiscriminator`

Uses multiple subperiod discriminators for temporal discrimination.

---

## Loss Functions

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/losses.py`

**Discriminator Loss** (line 712):

```python
loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
```

**Generator Losses** (lines 722-726):

```python
loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0  # Mel-spectrogram
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl  # VAE KL divergence
loss_fm = feature_loss(fmap_r, fmap_g)  # Feature matching
loss_gen, _ = generator_loss(y_d_hat_g)  # Adversarial
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl  # Combined
```

---

## Optimizers & Scheduling

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/train.py`

**Optimizer Selection** (lines 60, 422-438):

```python
optimizer = "AdamW"  # Global variable

if optimizer == "AdamW":
    optimizer = torch.optim.AdamW
elif optimizer == "RAdam":
    optimizer = torch.optim.RAdam

optim_g = optimizer(
    net_g.parameters(),
    config.train.learning_rate * g_lr_coeff,  # Global variable
    betas=config.train.betas,
    eps=config.train.eps,
)
```

**Learning Rate Scheduling** (lines 503-512):

```python
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optim_g,
    gamma=config.train.lr_decay,
    last_epoch=epoch_str - 2,
)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
    optim_d,
    gamma=config.train.lr_decay,
    last_epoch=epoch_str - 2,
)

# Stepped each epoch (lines 778-779)
scheduler_d.step()
scheduler_g.step()
```

---

## Checkpoint System

**Saving** (lines 934-952):

```python
save_checkpoint(
    net_g,
    optim_g,
    config.train.learning_rate,
    epoch,
    lowest_g_value,
    consecutive_increases_gen,
    os.path.join(experiment_dir, f"G_{idx}.pth"),
)
```

**Loading** (`/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/utils.py`, lines 92-150):

```python
def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    checkpoint_dict = torch.load(
        checkpoint_path,
        map_location="cpu",  # CPU-safe loading
        weights_only=False,
    )

    # State dict handling with key replacement for compatibility
    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    new_state_dict = {
        k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()
    }

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))
```

---

## Data Loading

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/train/data_utils.py`

**Dataset** (lines 12-160):

```python
class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    def __init__(self, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        # Load metadata for each sample

    def __getitem__(self, index):
        # Return (spec, wav, phone, pitch, pitchf, speaker_id)
```

**DataLoader Setup** (lines 349-358):

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

**Bucket Sampler** (line 343):

```python
train_sampler = DistributedBucketSampler(
    train_dataset,
    batch_size * n_gpus,
    [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],  # Length buckets
    num_replicas=n_gpus,
    rank=rank,
    shuffle=True,
)
```

---

## Environment Setup

**Multiprocessing** (line 56):

```python
torch.multiprocessing.set_start_method("spawn", force=True)
```

**CUDA Backends** (lines 54-57):

```python
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method("spawn", force=True)
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
```

**Distributed Environment** (lines 168-170):

```python
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(randint(20000, 55555))
logger.info("MASTER_PORT: %s", os.environ["MASTER_PORT"])
```

---

## Monitoring & Logging

**TensorBoard Writer** (lines 313-316):

```python
if rank == 0:
    writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))
else:
    writer_eval = None
```

**Loss Tracking** (lines 69-77):

```python
avg_losses = {
    "grad_d_50": deque(maxlen=50),
    "grad_g_50": deque(maxlen=50),
    "disc_loss_50": deque(maxlen=50),
    "fm_loss_50": deque(maxlen=50),
    "kl_loss_50": deque(maxlen=50),
    "mel_loss_50": deque(maxlen=50),
    "gen_loss_50": deque(maxlen=50),
}
```

**Logging** (lines 744-773):

```python
if rank == 0 and global_step % 50 == 0:
    scalar_dict = {
        "grad_avg_50/norm_d": sum(avg_losses["grad_d_50"]) / len(avg_losses["grad_d_50"]),
        # ... more scalars
    }
    summarize(
        writer=writer,
        global_step=global_step,
        scalars=scalar_dict,
    )
```

---

## Special Features

### Overtraining Detection

**Lines 872-899**:

```python
if overtraining_detector:
    overtrain_info = f"Average epoch generator loss {avg_global_gen_loss:.3f}..."

    remaining_epochs_gen = max(
        overtraining_threshold - consecutive_increases_gen,
        0,
    )
    remaining_epochs_disc = max(
        overtraining_threshold * 2 - consecutive_increases_disc,
        0,
    )

    if remaining_epochs_disc == 0 or remaining_epochs_gen == 0:
        # Early stop
        done = True
```

### Gradient Checkpointing

**Model Parameter** (line 406):

```python
checkpointing=checkpointing,  # Boolean flag
```

Reduces memory by not storing activations, recomputes them during backward.

### AMD GPU Support (ZLUDA)

**File**: `/home/june/new/ultimate-rvc/src/ultimate_rvc/rvc/lib/zluda.py`

```python
if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):
    # Custom STFT implementation for compatibility
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
```

---

## Known Limitations

### No Device Type Support

- Hard-coded to CUDA or CPU
- No ROCm explicit support (uses ZLUDA as workaround)
- No Apple Silicon MPS optimization

### No Mixed Precision

- All computations in float32
- No automatic loss scaling
- Higher memory usage on modern GPUs

### No Gradient Accumulation

- Cannot simulate larger batch sizes
- Limited by GPU VRAM for small GPUs

### No Distributed Inference

- Only single-GPU inference
- No batch inference optimizations

### Device Transfer Overhead

- Manual transfers throughout code
- No optimization for transfer operations
- Repeated transfers in inference loop

---

## Recommended Integration Points for Accelerate

1. **Main training loop** (`run_training()` → `train_main()` → `train_and_evaluate()`)
2. **Model initialization** (net_g, net_d setup)
3. **Data loading** (DataLoader preparation)
4. **Config system** (Device detection)
5. **Checkpoint handling** (Device-safe loading/saving)
6. **Inference pipeline** (Optional mixed precision)
