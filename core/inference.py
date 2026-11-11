"""Research pipeline; individual explanation failures never fabricate output."""
import logging
import math
import time
from core.config import THRESHOLD
from core.precision import precision_context
from core.preprocessing import prepare_inputs

logger = logging.getLogger(__name__)

def analyze(bundle, image, ocr, caption, topics):
    import torch
    from xai.cross_attention import cross_attention
    from xai.incongruity import incongruity
    from xai.attention_rollout import attention_rollout
    from xai.integrated_gradients import integrated_gradients_text_batch
    from xai.visualization import heatmap

    t0 = time.perf_counter()
    with bundle.lock:
        inputs = prepare_inputs(image, ocr, caption, topics, bundle)
        model = bundle.model
        with torch.no_grad(), precision_context(model):
            logits, diagnostics = model(**inputs, return_diagnostics=True)
            probability = float(torch.softmax(logits.float(), dim=-1)[0, 1])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError('Invalid model probability')

        gate_ca = None
        if 'gate_values' in diagnostics and diagnostics['gate_values'] is not None:
            g_val = float(diagnostics['gate_values'].float().mean().item())
            if math.isfinite(g_val):
                gate_ca = max(0.0, min(1.0, g_val))

        explanations = {}
        patch_peaks = {}
        for name, method in [('cross_attention', lambda: cross_attention(diagnostics)),
                             ('incongruity', lambda: incongruity(diagnostics)),
                             ('rollout', lambda: attention_rollout(model, inputs['pixel_values']))]:
            try:
                scores = method().detach().cpu().numpy()
                patch_peaks[name] = float(scores.max())
                explanations[name] = heatmap(image, scores)
            except Exception:
                logger.exception('Explanation unavailable: %s', name)
                explanations[name] = None

        try:
            with torch.no_grad(), precision_context(model):
                vision = model.vision_encoder(pixel_values=inputs['pixel_values'], return_dict=True).last_hidden_state.detach()
            ig = integrated_gradients_text_batch(model, inputs['input_ids'], inputs['attention_mask'],
                inputs['topic_features'], vision, logits.argmax(dim=-1).detach(), steps=50, step_chunk=1)
            mask = model.semantic_text_mask(inputs['input_ids'], inputs['attention_mask'])[0]
            scores = ig['signed'][0].masked_fill(~mask, 0).cpu().tolist()
            if not all(math.isfinite(value) for value in scores):
                raise ValueError('Non-finite IG')
            tokens = bundle.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().tolist())
            explanations['ig'] = [(token, score) for token, score, keep in zip(tokens, scores, mask.cpu().tolist()) if keep]
        except Exception:
            logger.exception('Integrated Gradients unavailable')
            explanations['ig'] = None

        elapsed = time.perf_counter() - t0
        metrics = {
            'gate_ca': gate_ca,
            'gate_inc': (1.0 - gate_ca) if gate_ca is not None else None,
            'patch_peaks': patch_peaks,
            'elapsed_seconds': elapsed,
            'device': str(bundle.device),
        }
        return {
            'label': 'Hate' if probability >= THRESHOLD else 'Not Hate',
            'probability': probability,
            'xai': explanations,
            'metrics': metrics,
        }
