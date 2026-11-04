"""Explicit parameter scopes and deterministic, independent per-tensor noise."""
import hashlib
import torch


def tensor_seed(seed, name):
    payload = f"{int(seed)}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def generator_for(param, seed, name):
    return torch.Generator(device=param.device).manual_seed(tensor_seed(seed, name))


def select_parameters(model, scope, layer=None, part="layer"):
    if scope == "full":
        if layer is not None or part != "layer":
            raise ValueError("Full scope requires layer=None and part=layer")
        selected = {id(p) for p in model.parameters()}
    elif scope == "layer":
        layers = model.model.layers
        if type(layer) is not int or not 0 <= layer < len(layers):
            raise ValueError("Layer index is outside the model")
        block = layers[layer]
        if part == "attn":
            block = block.self_attn
        elif part == "mlp":
            block = block.mlp
        elif part != "layer":
            raise ValueError("Unknown layer part")
        selected = {id(p) for p in block.parameters()}
    else:
        raise ValueError("Scope must be full or layer")
    result = []
    for name, param in model.named_parameters():
        chosen = id(param) in selected
        if chosen and param.dtype not in (torch.float16, torch.float32):
            raise ValueError("Raw ZO requires float16 or float32 parameters")
        param.requires_grad_(chosen)
        if chosen:
            result.append((name, param))
    if not result:
        raise ValueError("Empty parameter scope")
    return result
