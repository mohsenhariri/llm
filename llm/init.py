import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    device: torch.device
    seed: int
    cache_dir: Path
    base_dir: Path


def init(seed: int = None) -> Config:
    """
    Initialize the environment settings for a machine learning project.

    Args:
        seed (int, optional): The seed for random number generators to ensure reproducibility. Defaults to None.

    Returns:
        Config: A frozen dataclass containing the configuration settings.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
        print("Device name:", torch.cuda.get_device_name(0))
        print("Device count:", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("CUDA is not available")

    # Set Hugging Face environment variables
    hf_telemetry = 1  # Set to 1 to disable telemetry
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = str(hf_telemetry)

    # Ensure required environment variables are set
    cs_bash = os.getenv("CS_BASH")
    cs_home = os.getenv("CS_HOME")
    if not cs_bash:
        raise EnvironmentError("Environment variable CS_BASH is not set")
    if not cs_home:
        raise EnvironmentError("Environment variable CS_HOME is not set")

    # Set Hugging Face token from environment script
    env_path = Path(cs_bash) / ".env.py"
    if env_path.is_file():
        with open(env_path, "r") as env_file:
            env_script = env_file.read()
            exec(env_script)
    else:
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    cache_dir = Path(cs_home) / ".cache/misc"

    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return Config(device=device, seed=seed, cache_dir=cache_dir, base_dir=cs_home)
