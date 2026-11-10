"""IG from 09a cell 18: 50 right-Riemann steps, predicted logit, PAD baseline."""
import torch
from core.precision import precision_context
from xai.reference import forward_c_from_embeddings

def integrated_gradients_text_batch(model, input_ids, attention_mask, topic_features, vision_tokens, target_classes, steps=50, step_chunk=10):
    batch_size = int(input_ids.shape[0])
    embedding_layer = model.text_encoder.get_input_embeddings()
    with torch.no_grad():
        input_word_embeddings = embedding_layer(input_ids).detach()
        if model.tokenizer.pad_token_id is not None:
            baseline_ids = torch.full_like(input_ids, int(model.tokenizer.pad_token_id))
            baseline_word_embeddings = embedding_layer(baseline_ids).detach()
        else:
            baseline_word_embeddings = torch.zeros_like(input_word_embeddings)
    delta = input_word_embeddings - baseline_word_embeddings
    gradient_sum = torch.zeros_like(input_word_embeddings, dtype=torch.float32)
    alpha_values = torch.arange(1, steps + 1, device=input_ids.device, dtype=torch.float32) / float(steps)
    for start in range(0, steps, step_chunk):
        alphas = alpha_values[start:start + step_chunk]
        chunk_size = int(len(alphas))
        interpolated = baseline_word_embeddings.unsqueeze(0) + alphas.view(chunk_size, 1, 1, 1) * delta.unsqueeze(0)
        interpolated = interpolated.reshape(chunk_size * batch_size, input_word_embeddings.shape[1], input_word_embeddings.shape[2]).detach().requires_grad_(True)
        repeated_input_ids = input_ids.unsqueeze(0).expand(chunk_size, -1, -1).reshape(chunk_size * batch_size, -1)
        repeated_attention_mask = attention_mask.unsqueeze(0).expand(chunk_size, -1, -1).reshape(chunk_size * batch_size, -1)
        repeated_topic = topic_features.unsqueeze(0).expand(chunk_size, -1, -1).reshape(chunk_size * batch_size, -1)
        repeated_vision = vision_tokens.unsqueeze(0).expand(chunk_size, -1, -1, -1).reshape(chunk_size * batch_size, vision_tokens.shape[1], vision_tokens.shape[2])
        repeated_targets = target_classes.unsqueeze(0).expand(chunk_size, -1).reshape(-1)
        with precision_context(model):
            logits = forward_c_from_embeddings(model=model, word_embeddings=interpolated, vision_tokens=repeated_vision, input_ids=repeated_input_ids, attention_mask=repeated_attention_mask, topic_features=repeated_topic)
            target_logits = logits.gather(1, repeated_targets.unsqueeze(1)).squeeze(1)
        gradients = torch.autograd.grad(outputs=target_logits.sum(), inputs=interpolated, retain_graph=False, create_graph=False, allow_unused=False)[0]
        gradients = gradients.float().reshape(chunk_size, batch_size, input_word_embeddings.shape[1], input_word_embeddings.shape[2])
        gradient_sum += gradients.sum(dim=0)
        del (interpolated, logits, target_logits, gradients, repeated_input_ids, repeated_attention_mask, repeated_topic, repeated_vision, repeated_targets)
    average_gradient = gradient_sum / float(steps)
    integrated = delta.float() * average_gradient
    signed_token_ig = integrated.sum(dim=-1)
    absolute_token_ig = signed_token_ig.abs()
    l1_token_ig = integrated.abs().sum(dim=-1)
    return {'signed': signed_token_ig.detach(), 'absolute': absolute_token_ig.detach(), 'l1': l1_token_ig.detach()}
