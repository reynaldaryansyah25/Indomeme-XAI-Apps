"""Local encoder snapshots tied to exact checkpoint revision metadata."""
import hashlib
import json
from pathlib import Path
from core.config import ROOT, ResearchUnavailable

def local_encoder(name, spec, expected_revision):
    folder = ROOT / 'models/encoders' / name / expected_revision
    manifest_path = folder / 'source_manifest.json'
    if not manifest_path.is_file():
        raise ResearchUnavailable(f'Encoder snapshot missing: {name}. Run python scripts/download_encoders.py')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['repo_id'] != spec['id'] or manifest['revision'] != expected_revision
            or not expected_revision.startswith(spec['revision'])):
        raise ResearchUnavailable(f'Encoder provenance mismatch: {name}')
    for entry in manifest['files']:
        path = folder / entry['rfilename']
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ResearchUnavailable('Invalid encoder manifest path')
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ResearchUnavailable(f'Encoder file missing/size mismatch: {path}')
        if 'lfs' in entry:
            digest = hashlib.sha256()
            expected = entry['lfs']['sha256']
        else:
            digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
            expected = entry['blobId']
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ResearchUnavailable(f'Encoder integrity mismatch: {path}')
    return {**spec, 'id': str(folder)}
