"""Forward/rollout helpers from 09a cell 16; CPU uses full precision."""
import math
import torch
from torch.nn import functional as F
from core.precision import precision_context

def forward_c_from_tokens(model, text_tokens, vision_tokens, input_ids, attention_mask, topic_features, need_attention_weights):
    text_cls = text_tokens[:, 0, :]
    image_cls = vision_tokens[:, 0, :]
    image_patches = vision_tokens[:, 1:, :]
    z_t2i, t2i_weights = model.cross_t2i(query=text_cls.unsqueeze(1), key=image_patches, value=image_patches, need_weights=need_attention_weights, average_attn_weights=False)
    z_t2i = z_t2i.squeeze(1)
    z_i2t, i2t_weights = model.cross_i2t(query=image_cls.unsqueeze(1), key=text_tokens, value=text_tokens, key_padding_mask=~attention_mask.bool(), need_weights=need_attention_weights, average_attn_weights=False)
    z_i2t = z_i2t.squeeze(1)
    z_ca = model.ca_norm(model.ca_projection(torch.cat([z_t2i, z_i2t], dim=-1)))
    semantic_mask = model.semantic_text_mask(input_ids, attention_mask)
    text_projected = F.normalize(model.text_incongruity_projection(text_tokens), p=2, dim=-1)
    image_projected = F.normalize(model.image_incongruity_projection(image_patches), p=2, dim=-1)
    similarity = torch.einsum('bth,bph->btp', text_projected, image_projected)
    similarity = similarity.masked_fill(~semantic_mask.unsqueeze(-1), -10000.0)
    max_similarity = similarity.max(dim=1).values
    mismatch = 1.0 - max_similarity
    incongruity_weights = torch.softmax(mismatch, dim=-1)
    z_inc = torch.einsum('bp,bph->bh', incongruity_weights, image_projected)
    gate_values = torch.sigmoid(model.gate(torch.cat([z_ca, z_inc], dim=-1)))
    representation = model.icam_norm(gate_values * z_ca + (1.0 - gate_values) * z_inc)
    topic_embedding = model.topic_projection(topic_features)
    classifier_input = torch.cat([representation, topic_embedding], dim=-1)
    logits = model.classifier(classifier_input)
    diagnostics = {'semantic_mask': semantic_mask, 'incongruity_weights': incongruity_weights, 'gate_values': gate_values}
    if need_attention_weights:
        diagnostics['t2i_attention'] = t2i_weights
        diagnostics['i2t_attention'] = i2t_weights
    return (logits, diagnostics)

def forward_c_from_embeddings(model, word_embeddings, vision_tokens, input_ids, attention_mask, topic_features):
    text_outputs = model.text_encoder(inputs_embeds=word_embeddings, attention_mask=attention_mask, return_dict=True)
    logits, _ = forward_c_from_tokens(model=model, text_tokens=text_outputs.last_hidden_state, vision_tokens=vision_tokens, input_ids=input_ids, attention_mask=attention_mask, topic_features=topic_features, need_attention_weights=False)
    return logits

def capture_vision_tokens_and_rollout(model, pixel_values):
    layer_inputs = [None for _ in model.vision_encoder.encoder.layer]
    handles = []
    for layer_index, layer in enumerate(model.vision_encoder.encoder.layer):
        attention_module = layer.attention.attention

        def make_hook(index):

            def hook(module, args):
                layer_inputs[index] = args[0].detach()
            return hook
        handles.append(attention_module.register_forward_pre_hook(make_hook(layer_index)))
    try:
        with torch.no_grad():
            with precision_context(model):
                vision_outputs = model.vision_encoder(pixel_values=pixel_values, return_dict=True)
        vision_tokens = vision_outputs.last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    if any((value is None for value in layer_inputs)):
        raise RuntimeError('Failed to capture all ViT self-attention inputs.')
    num_heads = int(model.vision_encoder.config.num_attention_heads)
    hidden_size = int(model.vision_encoder.config.hidden_size)
    head_dim = hidden_size // num_heads
    joint_attention = None
    for layer, hidden_states in zip(model.vision_encoder.encoder.layer, layer_inputs):
        attention_module = layer.attention.attention
        batch_size, sequence_length, _ = hidden_states.shape
        with torch.no_grad():
            with precision_context(model):
                queries = attention_module.query(hidden_states)
                keys = attention_module.key(hidden_states)
        queries = queries.view(batch_size, sequence_length, num_heads, head_dim).permute(0, 2, 1, 3).float()
        keys = keys.view(batch_size, sequence_length, num_heads, head_dim).permute(0, 2, 1, 3).float()
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(float(head_dim))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        mean_attention = attention_probs.mean(dim=1)
        identity = torch.eye(sequence_length, device=mean_attention.device, dtype=mean_attention.dtype).unsqueeze(0)
        augmented = mean_attention + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if joint_attention is None:
            joint_attention = augmented
        else:
            joint_attention = torch.bmm(augmented, joint_attention)
    rollout = joint_attention[:, 0, 1:]
    rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (vision_tokens.detach(), rollout.detach())
