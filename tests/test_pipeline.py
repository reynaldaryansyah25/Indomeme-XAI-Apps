import hashlib
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
from core.config import CHECKPOINT, CHECKPOINT_SHA, ResearchUnavailable, load_protocol
from core.preprocessing import InvalidImage, validate_image, format_controlled_text, topic_to_multihot

def picture(format='PNG'):
    out = BytesIO()
    Image.new('RGB', (24, 16), 'white').save(out, format=format)
    return out.getvalue()

class UploadTests(unittest.TestCase):
    def test_supported_formats_return_rgb(self):
        for format, suffix in [('JPEG', '.jpg'), ('JPEG', '.jpeg'), ('PNG', '.png'), ('WEBP', '.webp')]:
            with self.subTest(format=format, suffix=suffix):
                image = validate_image(picture(format), 'meme'+suffix)
                self.assertEqual(image.mode, 'RGB')
                self.assertEqual(image.size, (24, 16))

    def test_rejects_bad_images(self):
        for data, name in [(b'', 'a.png'), (b'bad', 'a.jpg'), (picture(), 'a.jpg'), (picture(), 'a.svg')]:
            with self.subTest(name=name), self.assertRaises(InvalidImage):
                validate_image(data, name)

    def test_size_and_pixel_limits(self):
        with patch('core.preprocessing.MAX_BYTES', 10), self.assertRaises(InvalidImage):
            validate_image(picture(), 'a.png')
        with patch('core.preprocessing.MAX_PIXELS', 100), self.assertRaises(InvalidImage):
            validate_image(picture(), 'a.png')

class ResearchContractTests(unittest.TestCase):
    def test_supplied_frozen_protocol_is_valid(self):
        protocol = load_protocol()
        self.assertEqual(protocol['models']['indobertweet']['max_length'], 128)
        self.assertEqual(protocol['dataset']['topic_categories'], [
            'political', 'national origin', 'individual', 'institutions',
            'gender', 'religion', 'social sub-groups', 'unknown'])

    def test_missing_protocol_stops_pipeline(self):
        with patch.dict('os.environ', {'PROTOCOL_PATH': 'does-not-exist.yaml'}):
            with self.assertRaises(ResearchUnavailable):
                load_protocol()

    def test_wrong_protocol_hash_stops_pipeline(self):
        with patch.dict('os.environ', {'PROTOCOL_PATH': __file__}):
            with self.assertRaises(ResearchUnavailable):
                load_protocol()

    def test_notebook_ocr_example_and_caption_preservation(self):
        text = format_controlled_text('GW gak liat #Kamu @abc https://x.com', 'CAPTION Asli', 'political')
        self.assertEqual(text, '[political] CAPTION Asli [SEP] saya tidak lihat kamu')

    def test_multihot_obeys_supplied_protocol_order(self):
        categories = ['unknown','religion','political','individual','gender','institutions','national origin','social sub-groups']
        self.assertEqual(topic_to_multihot(['political', 'religion'], categories).tolist(), [0,1,1,0,0,0,0,0])
        self.assertEqual(topic_to_multihot([], categories).tolist(), [1,0,0,0,0,0,0,0])
        with self.assertRaises(ValueError):
            topic_to_multihot(['invalid'], categories)

    def test_real_checkpoint_hash(self):
        digest = hashlib.sha256()
        with CHECKPOINT.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), CHECKPOINT_SHA)

class StreamlitTests(unittest.TestCase):
    def test_empty_app_has_no_prediction_and_disabled_analysis(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.tabs), 4)
        self.assertNotIn('analysis', app.session_state)

    def test_valid_upload_then_unavailable_model_shows_error_only(self):
        from streamlit.testing.v1 import AppTest
        upload = BytesIO(picture())
        upload.name = 'meme.png'
        with patch('streamlit.file_uploader', return_value=upload), patch.dict(
                'os.environ', {'PROTOCOL_PATH': 'does-not-exist.yaml'}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.button[1].disabled)
            app.button[1].click().run(timeout=20)
            self.assertEqual(len(app.exception), 0)
            self.assertNotIn('analysis', app.session_state)
            self.assertIn('Analisis belum dapat dijalankan', app.session_state['analysis_error'])

    def test_failed_model_load_does_not_return_prediction(self):
        # Validates the actual readiness guard; does not simulate model output.
        from models.loader import load_model
        with patch.dict('os.environ', {'PROTOCOL_PATH': 'does-not-exist.yaml'}):
            with self.assertRaises(ResearchUnavailable):
                load_model()

if __name__ == '__main__':
    unittest.main()
