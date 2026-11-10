"""Validated in-memory uploads and deterministic notebook preprocessing."""
from io import BytesIO
from pathlib import Path
import warnings
from PIL import Image, UnidentifiedImageError
from core.config import MAX_BYTES, MAX_PIXELS

class InvalidImage(ValueError):
    pass

def validate_image(data, filename):
    expected = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP'}
    suffix = Path(filename).suffix.lower()
    if suffix not in expected:
        raise InvalidImage('Gunakan gambar JPG, JPEG, PNG, atau WEBP.')
    if not data or len(data) > MAX_BYTES:
        raise InvalidImage('Ukuran gambar harus lebih dari 0 dan maksimal 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format != expected[suffix]:
                    raise InvalidImage('Isi gambar tidak sesuai dengan format berkas.')
                if image.width * image.height > MAX_PIXELS:
                    raise InvalidImage('Resolusi gambar terlalu besar. Gunakan maksimal 20 megapiksel.')
                if getattr(image, 'n_frames', 1) != 1:
                    raise InvalidImage('Gunakan gambar statis tanpa animasi.')
                image.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()
                return image.convert('RGB')
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidImage('Gambar rusak atau tidak dapat dibaca. Pilih berkas lain.') from exc

def topic_to_multihot(topics, categories):
    import numpy as np
    if len(categories) != 8 or len(set(categories)) != 8:
        raise ValueError('Eight ordered research categories required')
    vector = np.zeros(8, dtype=np.float32)
    for topic in topics or ['unknown']:
        if topic not in categories:
            raise ValueError(f'Unknown topic: {topic}')
        vector[categories.index(topic)] = 1.0
    return vector

def format_controlled_text(ocr, caption, topic_text):
    from core.text import safe_text, clean_ocr
    return f'[{safe_text(topic_text).lower() or "unknown"}] {safe_text(caption)} [SEP] {clean_ocr(ocr)}'.strip()

def prepare_inputs(image, ocr, caption, topics, bundle):
    import torch
    from torchvision import transforms
    categories = bundle.protocol['dataset']['topic_categories']
    # Retain caller topic string order; model multihot uses protocol order.
    text = format_controlled_text(ocr, caption, ', '.join(topics or ['unknown']))
    encoded = bundle.tokenizer(text, padding='max_length', truncation=True,
        max_length=int(bundle.protocol['models']['indobertweet']['max_length']), return_tensors='pt')
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=bundle.processor.image_mean, std=bundle.processor.image_std),
    ])
    return {
        'input_ids': encoded['input_ids'].to(bundle.device),
        'attention_mask': encoded['attention_mask'].to(bundle.device),
        'pixel_values': transform(image.convert('RGB')).unsqueeze(0).to(bundle.device),
        'topic_features': torch.tensor(topic_to_multihot(topics, categories)).unsqueeze(0).to(bundle.device),
    }
