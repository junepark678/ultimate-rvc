from typing import TYPE_CHECKING

import json
import logging
import os
import pathlib

import torch

from ultimate_rvc.rvc.common import RVC_CONFIGS_DIR

if TYPE_CHECKING:
    from ultimate_rvc.rvc.lib.accelerate_utils import AccelerateConfig

logger = logging.getLogger(__name__)

version_config_paths = [
    os.path.join("48000.json"),
    os.path.join("40000.json"),
    os.path.join("32000.json"),
]


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    """
    Configuration manager for RVC models with device detection and
    Accelerate integration.

    This class implements a singleton pattern to manage RVC
    configurations including device selection, GPU memory detection,
    and optional Accelerate support. It supports environment variable
    overrides for device selection via URVC_ACCELERATOR.

    """

    def __init__(self) -> None:
        """
        Initialize Config with device detection and environment
        variable support.

        Checks URVC_ACCELERATOR environment variable for device
        override. Valid values are: 'auto', 'cuda', 'rocm', 'cpu'.
        If not set, uses automatic detection.

        """
        self._set_device()
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def _set_device(self) -> None:
        """
        Set device based on environment variable or auto-detection.

        Checks URVC_ACCELERATOR environment variable. If set to
        'cuda', 'rocm', or 'cpu', uses that device. Otherwise,
        uses automatic detection. Falls back to 'cpu' if no
        acceleration hardware is available.

        """
        accelerator_env = os.getenv("URVC_ACCELERATOR", "").lower()

        if accelerator_env == "cuda":
            # User explicitly requested CUDA
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.gpu_name = torch.cuda.get_device_name(0)
            else:
                logger.warning(
                    "URVC_ACCELERATOR set to 'cuda' but CUDA not "
                    "available. Falling back to CPU."
                )
                self.device = "cpu"
                self.gpu_name = None
        elif accelerator_env == "rocm":
            # User explicitly requested ROCm
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = "cpu"  # ROCm handled separately
                self.gpu_name = None
            else:
                logger.warning(
                    "URVC_ACCELERATOR set to 'rocm' but XPU not "
                    "available. Falling back to CPU."
                )
                self.device = "cpu"
                self.gpu_name = None
        elif accelerator_env == "cpu":
            # User explicitly requested CPU
            self.device = "cpu"
            self.gpu_name = None
        elif accelerator_env in ("auto", ""):
            # Use automatic detection
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.gpu_name = (
                torch.cuda.get_device_name(0)
                if self.device.startswith("cuda")
                else None
            )
        else:
            # Invalid value, log warning and use auto-detection
            logger.warning(
                "Invalid URVC_ACCELERATOR value: %s. Valid options "
                "are: auto, cuda, rocm, cpu. Using auto-detection.",
                accelerator_env,
            )
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.gpu_name = (
                torch.cuda.get_device_name(0)
                if self.device.startswith("cuda")
                else None
            )

    def load_config_json(self) -> dict:
        configs = {}
        for config_file in version_config_paths:
            config_path = os.path.join(str(RVC_CONFIGS_DIR), config_file)
            with pathlib.Path(config_path).open() as f:
                configs[config_file] = json.load(f)
        return configs

    def has_mps(self) -> bool:
        # Check if Metal Performance Shaders are available - for macOS 12.3+.
        return torch.backends.mps.is_available()

    def has_xpu(self) -> bool:
        # Check if XPU is available.
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def device_config(self) -> tuple:
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        elif self.has_mps():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self) -> None:
        """
        Configure CUDA settings for the detected GPU device.

        Sets GPU name and available memory in GB. Called automatically
        during device_config() if CUDA device is detected.

        """
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)

        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (
            1024**3
        )

    def use_accelerate(self) -> bool:
        """
        Check if Accelerate should be used for training.

        Returns True if Accelerate is available (installed).
        Availability is determined by attempting to import the
        Accelerate library.

        Returns
        -------
        bool
            True if Accelerate is available, False otherwise.

        """
        try:
            import accelerate  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False

    def get_accelerate_config(
        self,
    ) -> "AccelerateConfig | None":
        """
        Get Accelerate configuration based on current settings.

        Creates an AccelerateConfig instance with device type and
        other settings based on current Config state. If Accelerate
        is not available, logs a warning and returns None.

        Returns
        -------
        AccelerateConfig or None
            Accelerate configuration if available, None if Accelerate
            is not installed.

        """
        try:

            from ultimate_rvc.rvc.lib.accelerate_utils import (
                AccelerateConfig,
            )

            # Determine device type from current device setting
            device_type = "auto"
            if self.device == "cpu":
                device_type = "cpu"
            elif self.device.startswith("cuda"):
                device_type = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_type = "rocm"

            return AccelerateConfig(device_type=device_type)
        except ImportError:
            logger.warning(
                "Accelerate is not installed. Cannot create "
                "AccelerateConfig. Install with: pip install accelerate"
            )
            return None


def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb
    return "8"


def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4,
            )
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    if len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = (
            "Unfortunately, there is no compatible GPU available to support your"
            " training."
        )
    return gpu_info


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    return "-"
