"""Automatic discovery and fetching of frozen research artifacts for cloud deployment."""
import hashlib
import logging
from pathlib import Path
import urllib.request
from core.config import CHECKPOINT, CHECKPOINT_SHA, ROOT

logger = logging.getLogger(__name__)

CHECKPOINT_URL = 'https://github.com/reynaldaryansyah25/Indomeme-XAI-Apps/releases/download/v1.0.0/best_model.pt'

def verify_file_sha256(path, expected_sha):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha

def ensure_checkpoint_available():
    """Ensure the verified frozen model checkpoint is available on disk."""
    if CHECKPOINT.is_file() and verify_file_sha256(CHECKPOINT, CHECKPOINT_SHA):
        return CHECKPOINT

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CHECKPOINT.with_suffix('.pt.partial')
    logger.info('Fetching representative checkpoint from GitHub Release asset...')
    print('Mengunduh checkpoint representatif ICAM (best_model.pt)...', flush=True)

    headers = {'User-Agent': 'IndoMeme-XAI-Deployment/1.0'}
    req = urllib.request.Request(CHECKPOINT_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp, temp_path.open('wb') as out:
        for chunk in iter(lambda: resp.read(8 * 1024 * 1024), b''):
            out.write(chunk)

    if not verify_file_sha256(temp_path, CHECKPOINT_SHA):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError('Downloaded checkpoint failed SHA-256 integrity verification.')

    temp_path.replace(CHECKPOINT)
    logger.info('Checkpoint successfully verified and placed at %s', CHECKPOINT)
    return CHECKPOINT

def ensure_encoders_available():
    """Ensure local encoder snapshots are downloaded and verified."""
    indobertweet_dir = ROOT / 'models/encoders/indobertweet'
    vit_dir = ROOT / 'models/encoders/vit'

    needs_download = False
    if not indobertweet_dir.exists() or not list(indobertweet_dir.rglob('source_manifest.json')):
        needs_download = True
    if not vit_dir.exists() or not list(vit_dir.rglob('source_manifest.json')):
        needs_download = True

    if needs_download:
        logger.info('Downloading encoder snapshots from Hugging Face...')
        print('Mengunduh snapshot encoder penelitian (IndoBERTweet & ViT)...', flush=True)
        from scripts.download_encoders import ENCODERS, download_encoder
        for item in ENCODERS.items():
            download_encoder(item)
        logger.info('Encoders successfully downloaded and verified.')

def ensure_environment_ready():
    """Ensure all frozen artifacts (checkpoint & encoders) are available before inference."""
    ensure_checkpoint_available()
    ensure_encoders_available()
