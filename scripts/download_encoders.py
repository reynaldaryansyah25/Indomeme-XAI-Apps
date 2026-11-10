"""Fetch only exact frozen encoder artifacts; no training or alternative revisions.

Uses the standard library so it can run before installing the ML runtime.
Every file is checked against the pinned Hub revision's LFS SHA-256/git blob ID.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {
    'indobertweet': ('indolem/indobertweet-base-uncased', 'c77ff8c6b42c1d8e357e17c0e1f83c798531beef',
        {'config.json', 'pytorch_model.bin', 'vocab.txt', 'added_tokens.json', 'special_tokens_map.json', 'tokenizer_config.json'}),
    'vit': ('google/vit-base-patch16-224-in21k', 'b4569560a39a0f1af58e3ddaf17facf20ab919b0',
        {'config.json', 'model.safetensors', 'preprocessor_config.json'}),
}

def request(url):
    return urlopen(Request(url, headers={'User-Agent': 'IndoMeme-XAI-setup/1.0'}), timeout=90)

def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    if 'lfs' in entry:
        digest = hashlib.sha256()
        expected = entry['lfs']['sha256']
    else:
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        expected = entry['blobId']
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest() == expected

def download_encoder(item):
    name, (repo, revision, filenames) = item
    with request(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
        info = json.load(response)
    if info['sha'] != revision:
        raise RuntimeError(f'Revision mismatch for {name}')
    entries = [entry for entry in info['siblings'] if entry['rfilename'] in filenames]
    if {entry['rfilename'] for entry in entries} != filenames:
        raise RuntimeError(f'Missing required files for {name}')
    destination = ROOT / 'models/encoders' / name / revision
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        filename = entry['rfilename']
        target = destination / filename
        if verify(target, entry):
            print(f'Verified {name}/{filename}', flush=True)
            continue
        temporary = target.with_name(filename + '.partial')
        for attempt in range(3):
            try:
                print(f'Downloading {name}/{filename} ({entry["size"]/1024/1024:.1f} MiB)', flush=True)
                with request(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}') as response, temporary.open('wb') as out:
                    done = 0
                    last = 0
                    while block := response.read(1024*1024):
                        out.write(block)
                        done += len(block)
                        if done-last >= 64*1024*1024:
                            print(f'{name}/{filename}: {done/1024/1024:.0f} / {entry["size"]/1024/1024:.0f} MiB', flush=True)
                            last = done
                if not verify(temporary, entry):
                    raise RuntimeError(f'Integrity check failed for {filename}')
                temporary.replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    manifest = {'repo_id': repo, 'revision': revision, 'files': entries}
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'READY {name}: {destination}', flush=True)

def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download_encoder, ENCODERS.items()))

if __name__ == '__main__':
    main()
