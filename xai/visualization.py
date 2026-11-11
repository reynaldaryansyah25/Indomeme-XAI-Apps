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
    # Smooth two-stop ramp: deep indigo (#6366f1) -> vibrant rose (#f43f5e) at peak focus
    t = np.clip(field[..., None], 0.0, 1.0)
    low_rgb = np.array([99.0, 102.0, 241.0], dtype=np.float32)
    high_rgb = np.array([244.0, 63.0, 94.0], dtype=np.float32)
    color = low_rgb * (1.0 - t) + high_rgb * t
    alpha = field[..., None] * 0.68
    return Image.fromarray(np.clip(base * (1.0 - alpha) + color * alpha, 0, 255).astype('uint8'))

def token_html(tokens):
    maximum = max((abs(score) for _, score in tokens), default=0) or 1
    container_style = (
        'display: flex; flex-wrap: wrap; gap: 6px; align-items: center; '
        'padding: 14px; background: rgba(15, 23, 42, 0.85); '
        'border: 1px solid rgba(148, 163, 184, 0.18); '
        'border-radius: 12px; width: 100%; max-width: 100%; box-sizing: border-box; '
        'overflow: hidden;'
    )
    chips = []
    for token, score in tokens:
        norm = abs(score) / maximum
        if score >= 0:
            alpha = 0.12 + 0.44 * norm
            bg = f'rgba(139, 92, 246, {alpha:.4f})'
            border = f'rgba(167, 139, 250, {0.22 + 0.48 * norm:.3f})'
        else:
            alpha = 0.10 + 0.36 * norm
            bg = f'rgba(16, 185, 129, {alpha:.4f})'
            border = f'rgba(52, 211, 153, {0.20 + 0.45 * norm:.3f})'

        chip_style = (
            f'background: {bg}; border: 1px solid {border}; '
            'display: inline-flex; align-items: center; gap: 5px; '
            'padding: 4px 9px; border-radius: 7px; '
            'font-size: 12px; font-family: "JetBrains Mono", ui-monospace, monospace; '
            'color: #f8fafc; white-space: normal; word-break: break-word; '
            'line-height: 1.35; box-sizing: border-box;'
        )
        chips.append(f'<span class="token-heat" title="IG: {score:+.5f}" style="{chip_style}">{html.escape(token)}</span>')
    return f'<div class="token-list" style="{container_style}">{" ".join(chips)}</div>'

def top_tokens_html(tokens, top_k=5):
    if not tokens:
        return ''
    ranked = sorted(tokens, key=lambda item: abs(item[1]), reverse=True)[:top_k]
    maximum = max((abs(score) for _, score in ranked), default=0) or 1
    rows = []
    for token, score in ranked:
        pct = max(4, min(100, int(round(abs(score) / maximum * 100))))
        is_pos = score >= 0
        bar_color = 'linear-gradient(90deg, #8b5cf6, #f43f5e)' if is_pos else 'linear-gradient(90deg, #10b981, #06b6d4)'
        val_color = '#c4b5fd' if is_pos else '#6ee7b7'
        rows.append(f'''
        <div class="ig-bar-row">
          <span class="ig-bar-token" title="{html.escape(token)}">{html.escape(token)}</span>
          <div class="ig-bar-track"><div class="ig-bar-fill" style="width:{pct}%;background:{bar_color};"></div></div>
          <span class="ig-bar-val" style="color:{val_color};">{score:+.4f}</span>
        </div>
        ''')
    return f'''
    <div class="ig-top-box">
      <div class="ig-top-title">
        <span>Top-{len(ranked)} Token Paling Berpengaruh</span>
        <span class="ig-top-legend"><i style="background:#8b5cf6;"></i> Positif (Hate) &nbsp;·&nbsp; <i style="background:#10b981;"></i> Negatif (Netral)</span>
      </div>
      {"".join(rows)}
    </div>
    '''
