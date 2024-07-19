import numpy as np
import torch


def rgb_to_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device) / 255.0
