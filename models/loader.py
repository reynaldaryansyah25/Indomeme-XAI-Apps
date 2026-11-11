"""Strict reconstruction; no replacement model and no random-weight fallback."""
from dataclasses import dataclass
import gc
import logging
import os
import threading
from core.config import (
    CHECKPOINT,
    CHECKPOINT_SHA,
    PROTOCOL_SHA,
    ResearchUnavailable,
    load_protocol,
    resolve_checkpoint_path,
    sha256,
)

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

    ckpt_path = resolve_checkpoint_path()
    if not ckpt_path.is_file() or sha256(ckpt_path) != CHECKPOINT_SHA:
        raise ResearchUnavailable('Checkpoint missing or SHA-256 mismatch')

    import torch
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    from models.icam import ICAM, assert_freeze_policy
    from core.encoders import local_encoder

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('checkpoint_format') != 'trainable_state_dict_v1'
        or checkpoint.get('config_hash') != PROTOCOL_SHA
        or checkpoint.get('variant') != 'C' or checkpoint.get('seed') != 999):
        raise ResearchUnavailable('Checkpoint metadata does not match frozen experiment')

    revisions = checkpoint['model_revisions']
    state = checkpoint.pop('trainable_state', None)
    if not isinstance(state, dict) or not state:
        raise ResearchUnavailable('Checkpoint trainable_state missing')
    del checkpoint
    gc.collect()

    text = local_encoder('indobertweet', protocol['models']['indobertweet'], revisions['indobertweet']['resolved_revision'])
    vision = local_encoder('vit', protocol['models']['vit'], revisions['vit']['resolved_revision'])
    tokenizer = AutoTokenizer.from_pretrained(text['id'], local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(vision['id'], local_files_only=True)

    try:
        model = ICAM.build_from_checkpoint_state(text, vision, tokenizer, state, local_files_only=True)
    except KeyError as exc:
        raise ResearchUnavailable('Trainable checkpoint parameter keys mismatch') from exc
    finally:
        state.clear()
        del state
        gc.collect()

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
