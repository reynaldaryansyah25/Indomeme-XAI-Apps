"""Automatic discovery, verification, and fetching of frozen research artifacts for cloud deployment."""
import hashlib
import json
import logging
import time
import urllib.request
from core.config import (
    CHECKPOINT,
    CHECKPOINT_SHA,
    ROOT,
    resolve_checkpoint_path,
    writable_artifacts_root,
)

logger = logging.getLogger(__name__)

CHECKPOINT_URL = (
    'https://github.com/reynaldaryansyah25/Indomeme-XAI-Apps/releases/download/v1.0.0/best_model.pt'
)

ENCODERS = {
    'indobertweet': (
        'indolem/indobertweet-base-uncased',
        'c77ff8c6b42c1d8e357e17c0e1f83c798531beef',
        {
            'config.json',
            'pytorch_model.bin',
            'vocab.txt',
            'added_tokens.json',
            'special_tokens_map.json',
            'tokenizer_config.json',
        },
    ),
    'vit': (
        'google/vit-base-patch16-224-in21k',
        'b4569560a39a0f1af58e3ddaf17facf20ab919b0',
        {'config.json', 'model.safetensors', 'preprocessor_config.json'},
    ),
}

def _http_open(url, timeout=180):
    req = urllib.request.Request(url, headers={'User-Agent': 'IndoMeme-XAI-Deployment/1.0'})
    return urllib.request.urlopen(req, timeout=timeout)

def verify_file_sha256(path, expected_sha):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha

def _verify_hf_entry(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    if 'lfs' in entry:
        digest = hashlib.sha256()
        expected = entry['lfs']['sha256']
    else:
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        expected = entry['blobId']
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == expected

def _manifest_quick_ok(folder, expected_files):
    manifest_path = folder / 'source_manifest.json'
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entries = manifest.get('files', [])
        if {e['rfilename'] for e in entries} != set(expected_files):
            return False
        return all(
            (folder / e['rfilename']).is_file()
            and (folder / e['rfilename']).stat().st_size == e['size']
            for e in entries
        )
    except Exception:
        return False

def artifacts_present():
    """Fast non-blocking check whether checkpoint and encoder manifests exist locally."""
    ckpt = resolve_checkpoint_path()
    if not ckpt.is_file() or ckpt.stat().st_size < 100_000_000:
        return False
    cache_root = writable_artifacts_root()
    for name, (_, revision, filenames) in ENCODERS.items():
        primary = ROOT / 'models/encoders' / name / revision
        fallback = cache_root / 'models/encoders' / name / revision
        if not (_manifest_quick_ok(primary, filenames) or _manifest_quick_ok(fallback, filenames)):
            return False
    return True

def ensure_checkpoint_available(progress_callback=None):
    """Ensure the verified frozen model checkpoint is available on disk."""
    existing = resolve_checkpoint_path()
    if existing.is_file() and verify_file_sha256(existing, CHECKPOINT_SHA):
        return existing

    target = writable_artifacts_root() / 'models/checkpoints/best_model.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and verify_file_sha256(target, CHECKPOINT_SHA):
        return target

    temp_path = target.with_suffix('.pt.partial')
    msg = 'Mengunduh bobot checkpoint ICAM Varian C (~578 MiB) dari GitHub Release...'
    logger.info(msg)
    if progress_callback:
        progress_callback(msg)

    for attempt in range(3):
        try:
            temp_path.unlink(missing_ok=True)
            with _http_open(CHECKPOINT_URL, timeout=300) as resp, temp_path.open('wb') as out:
                total = int(resp.headers.get('Content-Length') or 0)
                downloaded = 0
                last_reported = 0
                for chunk in iter(lambda: resp.read(4 * 1024 * 1024), b''):
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and downloaded - last_reported >= 64 * 1024 * 1024:
                        mb = downloaded / (1024 * 1024)
                        tot_mb = total / (1024 * 1024) if total else 578
                        progress_callback(f'Mengunduh checkpoint ICAM: {mb:.0f} / {tot_mb:.0f} MiB...')
                        last_reported = downloaded

            if progress_callback:
                progress_callback('Memverifikasi integritas SHA-256 checkpoint ICAM...')
            if not verify_file_sha256(temp_path, CHECKPOINT_SHA):
                raise RuntimeError('Checkpoint yang diunduh gagal lolos verifikasi integritas SHA-256.')

            temp_path.replace(target)
            logger.info('Checkpoint verified at %s', target)
            return target
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            logger.warning('Checkpoint download attempt %d failed: %s', attempt + 1, exc)
            if attempt == 2:
                raise RuntimeError(f'Gagal mengunduh checkpoint model ({exc}).') from exc
            time.sleep(2 ** attempt)

def _download_single_encoder(name, repo, revision, filenames, base_root, progress_callback=None):
    destination = base_root / 'models/encoders' / name / revision
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'source_manifest.json'

    if _manifest_quick_ok(destination, filenames):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if all(_verify_hf_entry(destination / e['rfilename'], e) for e in manifest['files']):
            return destination

    msg = f'Menyingkronkan snapshot encoder {name} ({repo})...'
    logger.info(msg)
    if progress_callback:
        progress_callback(msg)

    info = None
    for attempt in range(3):
        try:
            with _http_open(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=90) as resp:
                info = json.load(resp)
            break
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f'Gagal mengakses metadata Hugging Face untuk {repo}: {exc}') from exc
            time.sleep(2 ** attempt)

    if info.get('sha') != revision:
        raise RuntimeError(f'Revision mismatch untuk {name}: diharapkan {revision}, diterima {info.get("sha")}')

    entries = [entry for entry in info['siblings'] if entry['rfilename'] in filenames]
    if {entry['rfilename'] for entry in entries} != set(filenames):
        raise RuntimeError(f'Berkas wajib tidak lengkap untuk encoder {name}')

    for entry in entries:
        filename = entry['rfilename']
        target = destination / filename
        if _verify_hf_entry(target, entry):
            continue

        temporary = target.with_name(filename + '.partial')
        size_mib = entry['size'] / (1024 * 1024)
        for attempt in range(3):
            try:
                temporary.unlink(missing_ok=True)
                if progress_callback and size_mib >= 10:
                    progress_callback(f'Mengunduh {name}/{filename} ({size_mib:.1f} MiB)...')
                with _http_open(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}', timeout=180) as resp, temporary.open('wb') as out:
                    done = 0
                    last = 0
                    for block in iter(lambda: resp.read(2 * 1024 * 1024), b''):
                        out.write(block)
                        done += len(block)
                        if progress_callback and done - last >= 64 * 1024 * 1024:
                            progress_callback(f'Mengunduh {name}/{filename}: {done / (1024 * 1024):.0f} / {size_mib:.0f} MiB...')
                            last = done
                if not _verify_hf_entry(temporary, entry):
                    raise RuntimeError(f'Pemeriksaan integritas gagal untuk {filename}')
                temporary.replace(target)
                break
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise RuntimeError(f'Gagal mengunduh {name}/{filename}: {exc}') from exc
                time.sleep(2 ** attempt)

    manifest = {'repo_id': repo, 'revision': revision, 'files': entries}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return destination

def ensure_encoders_available(progress_callback=None):
    """Ensure local encoder snapshots are downloaded and verified."""
    cache_root = writable_artifacts_root()
    for name, (repo, revision, filenames) in ENCODERS.items():
        primary = ROOT / 'models/encoders' / name / revision
        if _manifest_quick_ok(primary, filenames):
            manifest = json.loads((primary / 'source_manifest.json').read_text(encoding='utf-8'))
            if all(_verify_hf_entry(primary / e['rfilename'], e) for e in manifest['files']):
                continue
        _download_single_encoder(name, repo, revision, filenames, cache_root, progress_callback=progress_callback)

def ensure_environment_ready(progress_callback=None):
    """Ensure all frozen artifacts (checkpoint & encoders) are available before inference."""
    ensure_checkpoint_available(progress_callback=progress_callback)
    ensure_encoders_available(progress_callback=progress_callback)
