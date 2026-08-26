"""Model + tokenizer loading. Nothing experiment-specific lives here."""

import torch
from transformer_lens import HookedTransformer


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(model_name: str, device: str = "auto", dtype=torch.float32):
    """Returns (HookedTransformer, tokenizer, device).

    `no_processing` keeps the original weights: no LayerNorm folding or weight
    centring, so cached activations match a plain HF forward pass.
    """
    device = resolve_device(device)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=dtype
    )
    model.eval()
    return model, model.tokenizer, device
