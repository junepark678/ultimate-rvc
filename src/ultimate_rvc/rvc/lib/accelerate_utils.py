"""
Utilities for integrating Hugging Face Accelerate into training and
inference systems.

This module provides a thin wrapper around Accelerate to simplify
distributed training, mixed precision, and gradient accumulation
configuration. It supports both training and inference modes with
automatic device detection.
"""

from __future__ import annotations

from typing import Any

import logging
import os
from dataclasses import dataclass, field

import torch

try:
    from accelerate import Accelerator
except ImportError:
    # Accelerate is optional; lazy load in functions
    Accelerator = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class AccelerateConfig:
    """
    Configuration for Hugging Face Accelerate.

    This class manages environment-based configuration for Accelerate,
    supporting mixed precision training, gradient accumulation, and
    device selection. It provides sensible defaults while allowing
    environment variable overrides.

    Attributes
    ----------
    mixed_precision : str
        Mixed precision mode: 'none', 'fp16', or 'bf16'.
    grad_accumulation_steps : int
        Number of gradient accumulation steps (default: 1).
    max_grad_norm : float
        Maximum gradient norm for clipping (default: 1.0).
    fp16_inference : bool
        Enable FP16 for inference (default: False).
    device_type : str
        Device type: 'auto', 'cuda', 'rocm', or 'cpu'.
    use_ddp : bool
        Use Distributed Data Parallel (default: True).
    ddp_backend : str
        DDP backend: 'nccl' for GPU, 'gloo' for CPU (default: 'auto').

    """

    mixed_precision: str = field(default="none")
    grad_accumulation_steps: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    fp16_inference: bool = field(default=False)
    device_type: str = field(default="auto")
    use_ddp: bool = field(default=True)
    ddp_backend: str = field(default="auto")

    def __post_init__(self) -> None:
        """
        Validate configuration and apply environment variable
        overrides.
        """
        self._apply_env_overrides()
        self._validate_config()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Mixed precision override
        mixed_precision = os.getenv("URVC_MIXED_PRECISION")
        if mixed_precision:
            self.mixed_precision = mixed_precision

        # Gradient accumulation override
        grad_accum = os.getenv("URVC_GRAD_ACCUMULATION")
        if grad_accum:
            try:
                self.grad_accumulation_steps = int(grad_accum)
            except ValueError:
                logger.warning(
                    "Invalid URVC_GRAD_ACCUMULATION value: %s. Using default: %d",
                    grad_accum,
                    self.grad_accumulation_steps,
                )

        # Max grad norm override
        max_grad = os.getenv("URVC_MAX_GRAD_NORM")
        if max_grad:
            try:
                self.max_grad_norm = float(max_grad)
            except ValueError:
                logger.warning(
                    "Invalid URVC_MAX_GRAD_NORM value: %s. Using default: %f",
                    max_grad,
                    self.max_grad_norm,
                )

        # FP16 inference override
        fp16_inf = os.getenv("URVC_FP16_INFERENCE")
        if fp16_inf:
            self.fp16_inference = fp16_inf.lower() in {"true", "1", "yes"}

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate mixed precision
        valid_precisions = ("none", "fp16", "bf16")
        if self.mixed_precision not in valid_precisions:
            logger.warning(
                "Invalid mixed_precision: %s. Using 'none'. Valid options: %s",
                self.mixed_precision,
                valid_precisions,
            )
            self.mixed_precision = "none"

        # Validate gradient accumulation steps
        if self.grad_accumulation_steps < 1:
            logger.warning(
                "grad_accumulation_steps must be >= 1, got %d. Using default: 1",
                self.grad_accumulation_steps,
            )
            self.grad_accumulation_steps = 1

        # Validate max grad norm
        if self.max_grad_norm <= 0:
            logger.warning(
                "max_grad_norm must be > 0, got %f. Using default: 1.0",
                self.max_grad_norm,
            )
            self.max_grad_norm = 1.0

        # Validate device type
        valid_devices = ("auto", "cuda", "rocm", "cpu")
        if self.device_type not in valid_devices:
            logger.warning(
                "Invalid device_type: %s. Using 'auto'. Valid options: %s",
                self.device_type,
                valid_devices,
            )
            self.device_type = "auto"

    @classmethod
    def from_env(cls) -> AccelerateConfig:
        """
        Create AccelerateConfig from environment variables.

        Returns
        -------
        AccelerateConfig
            Configuration instance populated from environment variables.

        """
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict[str, Any]
            Configuration as a dictionary.

        """
        return {
            "mixed_precision": self.mixed_precision,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "fp16_inference": self.fp16_inference,
            "device_type": self.device_type,
            "use_ddp": self.use_ddp,
            "ddp_backend": self.ddp_backend,
        }


def _detect_device_type() -> str:
    """
    Auto-detect available device type.

    Returns
    -------
    str
        Device type: 'cuda', 'rocm', or 'cpu'.

    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "rocm"
    return "cpu"


def _resolve_device_type(device_type: str) -> str:
    """
    Resolve device type string to actual device.

    Parameters
    ----------
    device_type : str
        Device type: 'auto', 'cuda', 'rocm', or 'cpu'.

    Returns
    -------
    str
        Resolved device type.

    """
    if device_type == "auto":
        return _detect_device_type()
    return device_type


def create_training_accelerator(
    config: AccelerateConfig | None = None,
    split_batches: bool = False,
) -> Accelerator:
    """
    Create Accelerator instance for training.

    This function initializes an Accelerator with optimal settings
    for distributed training, mixed precision, and gradient
    accumulation. It handles both single-GPU and multi-GPU training.

    Parameters
    ----------
    config : AccelerateConfig, optional
        Accelerate configuration. If None, created from environment.
    split_batches : bool, optional
        Whether to split batches across processes (default: False).

    Returns
    -------
    Accelerator
        Configured Accelerator instance for training.

    Raises
    ------
    ImportError
        If Accelerate is not installed.

    Examples
    --------
    >>> config = AccelerateConfig(mixed_precision="fp16")
    >>> accelerator = create_training_accelerator(config)
    >>> model, optimizer, train_loader = accelerator.prepare(
    ...     model, optimizer, train_loader
    ... )

    """
    if Accelerator is None:
        msg = "accelerate package is required for training"
        raise ImportError(msg)

    if config is None:
        config = AccelerateConfig.from_env()

    # Determine DDP backend
    ddp_backend = config.ddp_backend
    if ddp_backend == "auto":
        device_type = _resolve_device_type(config.device_type)
        ddp_backend = "nccl" if device_type == "cuda" else "gloo"

    # Create accelerator with training-specific settings
    mixed_precision = (
        config.mixed_precision if config.mixed_precision != "none" else None
    )
    ddp_backend_arg = ddp_backend if config.use_ddp else None
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=(config.grad_accumulation_steps),
        log_with="tensorboard",
        project_dir="runs",
        split_batches=split_batches,
        backend=ddp_backend_arg,
    )

    logger.info(
        "Training Accelerator created: mixed_precision=%s, grad_accum_steps=%d",
        config.mixed_precision,
        config.grad_accumulation_steps,
    )

    return accelerator


def create_inference_accelerator(
    config: AccelerateConfig | None = None,
) -> Accelerator:
    """
    Create Accelerator instance for inference.

    This function creates a lightweight Accelerator for inference with
    optional mixed precision support but no distributed training
    features.

    Parameters
    ----------
    config : AccelerateConfig, optional
        Accelerate configuration. If None, created from environment.

    Returns
    -------
    Accelerator
        Configured Accelerator instance for inference.

    Raises
    ------
    ImportError
        If Accelerate is not installed.

    Examples
    --------
    >>> config = AccelerateConfig(fp16_inference=True)
    >>> accelerator = create_inference_accelerator(config)
    >>> model = accelerator.prepare(model)

    """
    if Accelerator is None:
        msg = "accelerate package is required for inference"
        raise ImportError(msg)

    if config is None:
        config = AccelerateConfig.from_env()

    # Determine mixed precision for inference
    mixed_precision = None
    if config.fp16_inference:
        mixed_precision = "fp16"

    # Create lightweight accelerator for inference
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with=None,
    )

    logger.info(
        "Inference Accelerator created: mixed_precision=%s",
        mixed_precision,
    )

    return accelerator


def prepare_training_components(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader | None = None,
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader | None,
]:
    """
    Prepare training components with Accelerator.

    This function wraps model, optimizer, and dataloaders with the
    accelerator for distributed training and mixed precision.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator instance from create_training_accelerator().
    model : torch.nn.Module
        Model to prepare.
    optimizer : torch.optim.Optimizer
        Optimizer to prepare.
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader.
    eval_dataloader : torch.utils.data.DataLoader, optional
        Evaluation dataloader (default: None).

    Returns
    -------
    tuple
        Tuple of (model, optimizer, train_dataloader,
        eval_dataloader) prepared with accelerator.

    Examples
    --------
    >>> model, optimizer, train_loader, eval_loader = prepare_training_components(
    ...     accelerator, model, optimizer, train_loader, eval_loader
    ... )

    """
    # Prepare with accelerator
    prepared = accelerator.prepare(model, optimizer, train_dataloader)

    if eval_dataloader is not None:
        prepared_eval = accelerator.prepare(eval_dataloader)
        return (
            prepared[0],
            prepared[1],
            prepared[2],
            prepared_eval,
        )

    return prepared[0], prepared[1], prepared[2], None


def prepare_inference_model(
    accelerator: Accelerator,
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Prepare model for inference with Accelerator.

    This function wraps a model with the accelerator for inference
    with optional mixed precision.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator instance from create_inference_accelerator().
    model : torch.nn.Module
        Model to prepare.

    Returns
    -------
    torch.nn.Module
        Model prepared with accelerator.

    Examples
    --------
    >>> model = prepare_inference_model(accelerator, model)
    >>> with torch.no_grad():
    ...     output = model(input)

    """
    return accelerator.prepare(model)


def get_device_from_accelerator(
    accelerator: Accelerator,
) -> torch.device:
    """
    Get torch.device from Accelerator.

    This function extracts the device that the accelerator is using
    for training or inference.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator instance.

    Returns
    -------
    torch.device
        Device used by the accelerator.

    Examples
    --------
    >>> device = get_device_from_accelerator(accelerator)
    >>> tensor = torch.randn(10, device=device)

    """
    if hasattr(accelerator, "device"):
        return accelerator.device
    # Fallback: get device from accelerator's state
    if hasattr(accelerator.state, "device"):
        return accelerator.state.device
    # Final fallback: use first GPU or CPU
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def should_use_accelerate() -> bool:
    """
    Determine if Accelerate should be used.

    This checks if Accelerate is available and environment variables
    indicate it should be used.

    Returns
    -------
    bool
        True if Accelerate should be used, False otherwise.

    """
    return Accelerator is not None


def get_accelerator_config() -> AccelerateConfig:
    """
    Get current Accelerate configuration from environment.

    Returns
    -------
    AccelerateConfig
        Configuration from environment variables.

    """
    return AccelerateConfig.from_env()
