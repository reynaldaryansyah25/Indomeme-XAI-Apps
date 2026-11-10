from contextlib import nullcontext

def precision_context(model):
    import torch
    if next(model.parameters()).device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()
