"""Strict reconstruction; no replacement model and no random-weight fallback."""
from dataclasses import dataclass
import logging
import os
import threading
from core.config import CHECKPOINT, CHECKPOINT_SHA, PROTOCOL_SHA, ResearchUnavailable, load_protocol, sha256

@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    processor: object
    protocol: dict
    device: object
    lock: object

def load_model():
    protocol = load_protocol()
    from core.bootstrap import ensure_environment_ready
    ensure_environment_ready()

    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ResearchUnavailable('Checkpoint missing or SHA-256 mismatch')
    import torch
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    from models.icam import ICAM, assert_freeze_policy
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if (checkpoint.get('checkpoint_format') != 'trainable_state_dict_v1'
        or checkpoint.get('config_hash') != PROTOCOL_SHA
        or checkpoint.get('variant') != 'C' or checkpoint.get('seed') != 999):
        raise ResearchUnavailable('Checkpoint metadata does not match frozen experiment')
    from core.encoders import local_encoder
    revisions = checkpoint['model_revisions']
    text = local_encoder('indobertweet', protocol['models']['indobertweet'], revisions['indobertweet']['resolved_revision'])
    vision = local_encoder('vit', protocol['models']['vit'], revisions['vit']['resolved_revision'])
    tokenizer = AutoTokenizer.from_pretrained(text['id'], local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision['id'], local_files_only=True)
    model = ICAM(text, vision, tokenizer, local_files_only=True)
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    state = checkpoint['trainable_state']
    if expected != set(state):
        raise ResearchUnavailable('Trainable checkpoint parameter keys mismatch')
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)
    assert_freeze_policy(model)
    requested = os.environ.get('INDOMEME_DEVICE', 'auto')
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ResearchUnavailable('INDOMEME_DEVICE must be auto, cpu, or cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ResearchUnavailable('CUDA requested but unavailable')
    device = torch.device('cuda:0' if requested != 'cpu' and torch.cuda.is_available() else 'cpu')
    try:
        model.to(device).eval()
    except torch.cuda.OutOfMemoryError:
        if requested != 'auto':
            raise
        logging.getLogger(__name__).warning('GPU memory unavailable; loading the same model on CPU')
        model.cpu().eval()
        torch.cuda.empty_cache()
        device = torch.device('cpu')
    return ModelBundle(model, tokenizer, processor, protocol, device, threading.RLock())
