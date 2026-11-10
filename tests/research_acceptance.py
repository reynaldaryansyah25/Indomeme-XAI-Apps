"""Real checkpoint acceptance probe. No mock model, labels, probabilities or XAI.

This is a functional test on a drawn fixture, NOT an official-test parity claim.
"""
import argparse
import ast
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def smoke_image():
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (480, 360), '#d9efff')
    draw = ImageDraw.Draw(image)
    draw.ellipse((185, 80, 295, 190), fill='#ffd15c')
    draw.rectangle((0, 255, 480, 360), fill='#5da77b')
    draw.text((45, 25), 'SELAMAT PAGI', fill='#171d32', font_size=32)
    draw.text((30, 290), 'Semoga harimu menyenangkan', fill='white', font_size=24)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'docs/runtime_check.json'))
    args = parser.parse_args()
    import torch
    from core.config import CHECKPOINT, PROTOCOL_SHA, sha256
    from core.preprocessing import prepare_inputs, format_controlled_text
    from core.inference import analyze
    from core.precision import precision_context
    from models.loader import load_model
    from xai.reference import forward_c_from_tokens, capture_vision_tokens_and_rollout
    start = time.perf_counter()
    print('Loading real encoders and checkpoint...', flush=True)
    bundle = load_model()
    model = bundle.model
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable == 151415842, trainable
    image = smoke_image()
    ocr = 'SELAMAT PAGI Semoga harimu menyenangkan'
    caption = 'Ilustrasi matahari kuning di atas langit biru dan tanah hijau.'
    topics = ['unknown']
    inputs = prepare_inputs(image, ocr, caption, topics, bundle)
    assert tuple(inputs['input_ids'].shape) == (1, 128)
    assert tuple(inputs['pixel_values'].shape) == (1, 3, 224, 224)
    assert tuple(inputs['topic_features'].shape) == (1, 8)
    assert torch.isfinite(inputs['pixel_values']).all()
    # Execute only audited model definitions from the source, not notebook I/O/training.
    notebook = json.loads((ROOT / 'research/reference_notebooks/09a-v3-xai-representative-c.ipynb').read_text(encoding='utf-8'))
    definitions = ast.parse(''.join(notebook['cells'][13]['source']))
    scope = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'tokenizer': bundle.tokenizer}
    exec(compile(definitions, '<reference-notebook-cell-13>', 'exec'), scope)
    reference = scope['ControlledMultimodalModel']
    assert torch.equal(model.semantic_text_mask(inputs['input_ids'], inputs['attention_mask']),
        reference.semantic_text_mask(model, inputs['input_ids'], inputs['attention_mask']))
    with torch.no_grad(), precision_context(model):
        native, diagnostics = model(**inputs, return_diagnostics=True)
        reference_logits, reference_diagnostics = reference.forward(model, **inputs, return_diagnostics=True)
        text_tokens = model.text_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True).last_hidden_state
        vision_tokens = model.vision_encoder(pixel_values=inputs['pixel_values'], return_dict=True).last_hidden_state
        manual, _ = forward_c_from_tokens(model, text_tokens, vision_tokens, inputs['input_ids'], inputs['attention_mask'], inputs['topic_features'], True)
    exact_difference = float((native.float()-reference_logits.float()).abs().max())
    manual_difference = float((torch.softmax(native.float(), -1)-torch.softmax(manual.float(), -1)).abs().max())
    assert exact_difference == 0, exact_difference
    assert manual_difference <= 5e-4, manual_difference
    with torch.no_grad():
        _, rollout = capture_vision_tokens_and_rollout(model, inputs['pixel_values'])
    patch_maps = {
        'cross_attention': diagnostics['t2i_attention'].squeeze(2).float().mean(dim=1)[0],
        'rollout': rollout[0],
        'incongruity': diagnostics['incongruity_weights'].float()[0],
    }
    masses = {}
    for name, values in patch_maps.items():
        assert tuple(values.shape) == (196,), (name, values.shape)
        assert torch.isfinite(values).all() and (values >= 0).all(), name
        masses[name] = float(values.sum())
        assert abs(masses[name]-1) <= 1e-3, (name, masses[name])
    if bundle.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    print('Running full prediction and all four explanations (IG: 50 steps)...', flush=True)
    prediction_start = time.perf_counter()
    result = analyze(bundle, image, ocr, caption, topics)
    assert all(result['xai'].get(name) is not None for name in ['ig', 'cross_attention', 'rollout', 'incongruity'])
    assert result['xai']['ig'] and all(math.isfinite(score) for _, score in result['xai']['ig'])
    assert 0 <= result['probability'] <= 1
    assert abs(result['probability']-float(torch.softmax(native.float(), -1)[0, 1])) <= 5e-4
    assert result['label'] == ('Hate' if result['probability'] >= .45 else 'Not Hate')
    inference_seconds = time.perf_counter()-prediction_start
    peak_cuda = torch.cuda.max_memory_allocated()/1024**2 if bundle.device.type == 'cuda' else None
    cpu_probability = result['probability']
    if bundle.device.type == 'cuda':
        print('Checking the same loaded model on CPU...', flush=True)
        model.cpu()
        cpu_inputs = {key: value.cpu() for key, value in inputs.items()}
        with torch.no_grad():
            cpu_logits = model(**cpu_inputs)
            cpu_probability = float(torch.softmax(cpu_logits.float(), dim=-1)[0, 1])
        assert math.isfinite(cpu_probability) and 0 <= cpu_probability <= 1
    report = {
        'status': 'PASS', 'scope': 'real-model functional probe on a drawn test fixture',
        'official_notebook_probability_parity': 'NOT_TESTED: reference samples and official predictions unavailable',
        'environment': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'transformers', 'huggingface-hub', 'streamlit', 'numpy', 'pandas', 'Pillow']},
        'device': str(bundle.device), 'cuda_available': torch.cuda.is_available(),
        'checkpoint_sha256': sha256(CHECKPOINT), 'protocol_sha256': PROTOCOL_SHA,
        'trainable_parameter_count': trainable,
        'native_vs_source_notebook_logit_max_difference': exact_difference,
        'native_vs_manual_probability_max_difference': manual_difference,
        'patch_attribution_sums': masses,
        'smoke_input': {'ocr': ocr, 'caption': caption, 'topics': topics},
        'prediction': {'label': result['label'], 'hate_probability': result['probability']},
        'cpu_forward_probability': cpu_probability,
        'cpu_vs_primary_probability_difference': abs(cpu_probability-result['probability']),
        'xai_methods_passed': list(result['xai']), 'ig_steps': 50,
        'ig_semantic_token_count': len(result['xai']['ig']),
        'inference_and_xai_seconds': inference_seconds,
        'total_probe_seconds': time.perf_counter()-start,
        'peak_cuda_allocated_mib': peak_cuda,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
