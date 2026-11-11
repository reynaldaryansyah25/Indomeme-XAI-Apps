"""Internal research configuration. Never render these values in the UI."""
import hashlib
import os
from pathlib import Path
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'models/checkpoints/best_model.pt'
CHECKPOINT_SHA = 'fc76e1b4bf2288e99802356d63a8ad5345cef54732767c25c9e51178d21520c4'
PROTOCOL_SHA = 'd3ba1de81ef89a67946c43af996396d294d35f01a1640168bab18c8a3bec4f12'
THRESHOLD = 0.45
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
# Display order only. Model feature order MUST come from the verified protocol.
DISPLAY_TOPICS = ('political', 'national origin', 'institutions', 'individual',
                  'gender', 'social sub-groups', 'religion', 'unknown')

class ResearchUnavailable(RuntimeError):
    pass

def _is_dir_writable(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / '.write_probe'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False

def writable_artifacts_root():
    """Return ROOT if writable, otherwise a persistent/temp user cache folder."""
    override = os.environ.get('INDOMEME_CACHE_DIR')
    if override:
        candidate = Path(override)
        if _is_dir_writable(candidate):
            return candidate
    if _is_dir_writable(ROOT / 'models'):
        return ROOT
    home_cache = Path.home() / '.cache' / 'indomeme-xai'
    if _is_dir_writable(home_cache):
        return home_cache
    tmp_cache = Path(tempfile.gettempdir()) / 'indomeme-xai'
    tmp_cache.mkdir(parents=True, exist_ok=True)
    return tmp_cache

def resolve_checkpoint_path():
    """Locate existing verified checkpoint in ROOT first, then writable cache."""
    if CHECKPOINT.is_file():
        return CHECKPOINT
    fallback = writable_artifacts_root() / 'models/checkpoints/best_model.pt'
    return fallback

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_protocol():
    path = Path(os.environ.get('PROTOCOL_PATH', str(ROOT / 'research/frozen_artifacts/protocol_v3.yaml')))
    if not path.is_file():
        raise ResearchUnavailable(f'Missing frozen protocol: {path}')
    if sha256(path) != PROTOCOL_SHA:
        raise ResearchUnavailable('Frozen protocol SHA-256 mismatch')
    protocol = yaml.safe_load(path.read_text(encoding='utf-8'))
    if protocol['protocol']['status'] != 'FROZEN':
        raise ResearchUnavailable('Protocol is not frozen')
    categories = protocol['dataset']['topic_categories']
    if len(categories) != 8 or set(categories) != set(DISPLAY_TOPICS):
        raise ResearchUnavailable('Invalid frozen topic categories')
    return protocol
