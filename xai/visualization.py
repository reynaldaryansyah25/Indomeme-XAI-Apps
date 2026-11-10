"""Render only actual model scores; reject invalid or absent attributions."""
import html
import numpy as np
from PIL import Image

def heatmap(image, scores):
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (196,) or not np.isfinite(values).all() or values.min() < 0 or values.sum() <= 0:
        raise ValueError('Invalid patch attribution')
    spread = np.ptp(values)
    scaled = (values - values.min()) / spread if spread > 0 else np.zeros_like(values)
    # Limit rendering memory only; research input preprocessing remains unchanged.
    display = image.convert('RGB').copy()
    display.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    field = np.asarray(Image.fromarray(scaled.reshape(14, 14)).resize(display.size, Image.Resampling.BILINEAR))
    base = np.asarray(display, dtype=np.float32)
    color = np.empty_like(base)
    color[:] = [104, 77, 245]
    alpha = (field * 0.65)[..., None]
    return Image.fromarray(np.clip(base * (1-alpha) + color * alpha, 0, 255).astype('uint8'))

def token_html(tokens):
    maximum = max((abs(score) for _, score in tokens), default=0) or 1
    container_style = (
        'display: flex; flex-wrap: wrap; gap: 6px; align-items: center; '
        'padding: 12px 14px; background: #ffffff; border: 1px solid #e2e8f0; '
        'border-radius: 9px; width: 100%; max-width: 100%; box-sizing: border-box; '
        'overflow: hidden;'
    )
    chips = []
    for token, score in tokens:
        alpha = 0.06 + 0.34 * abs(score) / maximum
        chip_style = (
            f'background: rgba(104, 77, 245, {alpha:.4f}); '
            'display: inline-flex; align-items: center; '
            'padding: 4px 8px; border-radius: 5px; '
            'font-size: 12px; font-family: "JetBrains Mono", monospace; '
            'color: #0f172a; white-space: normal; word-break: break-word; '
            'line-height: 1.4; box-sizing: border-box;'
        )
        chips.append(f'<span class="token-heat" title="{score:.5f}" style="{chip_style}">{html.escape(token)}</span>')
    return f'<div class="token-list" style="{container_style}">{" ".join(chips)}</div>'
