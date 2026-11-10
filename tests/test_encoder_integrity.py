import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from core.encoders import local_encoder
from core.config import ResearchUnavailable

class SnapshotIntegrityTests(unittest.TestCase):
    def test_accepts_verified_file_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = 'a'*40
            folder = root / 'models/encoders/test' / revision
            folder.mkdir(parents=True)
            data = b'{}'
            (folder / 'config.json').write_bytes(data)
            manifest = {'repo_id': 'test/encoder', 'revision': revision, 'files': [
                {'rfilename': 'config.json', 'size': len(data),
                 'blobId': hashlib.sha1(b'blob 2\0'+data).hexdigest()}]}
            (folder / 'source_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            spec = {'id': 'test/encoder', 'revision': revision[:7]}
            with patch('core.encoders.ROOT', root):
                self.assertEqual(local_encoder('test', spec, revision)['id'], str(folder))
                (folder / 'config.json').write_bytes(b'[]')
                with self.assertRaises(ResearchUnavailable):
                    local_encoder('test', spec, revision)

    def test_missing_snapshot_does_not_use_another_revision(self):
        with tempfile.TemporaryDirectory() as directory, patch('core.encoders.ROOT', Path(directory)):
            with self.assertRaises(ResearchUnavailable):
                local_encoder('test', {'id': 'test/encoder', 'revision': 'a'*40}, 'a'*40)

if __name__ == '__main__':
    unittest.main()
