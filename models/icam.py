"""Architecture ported from reference notebook 09a, cell 13; not parity verified."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

HIDDEN_DIM = 768
NUM_HEADS = 8
ATTN_DROPOUT = 0.1
TOPIC_PROJ_DIM = 32

def set_module_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad = trainable

def freeze_controlled_backbones(text_encoder, vision_encoder):
    set_module_trainable(text_encoder.embeddings, False)
    for layer in text_encoder.encoder.layer[:2]:
        set_module_trainable(layer, False)
    set_module_trainable(vision_encoder.embeddings, False)
    for layer in vision_encoder.encoder.layer[:2]:
        set_module_trainable(layer, False)

def assert_freeze_policy(model):
    assert all((not parameter.requires_grad for parameter in model.text_encoder.embeddings.parameters()))
    for layer in model.text_encoder.encoder.layer[:2]:
        assert all((not parameter.requires_grad for parameter in layer.parameters()))
    assert any((parameter.requires_grad for parameter in model.text_encoder.encoder.layer[2].parameters()))
    assert all((not parameter.requires_grad for parameter in model.vision_encoder.embeddings.parameters()))
    for layer in model.vision_encoder.encoder.layer[:2]:
        assert all((not parameter.requires_grad for parameter in layer.parameters()))
    assert any((parameter.requires_grad for parameter in model.vision_encoder.encoder.layer[2].parameters()))

class ICAM(nn.Module):

    @classmethod
    def _init_fusion_heads(cls, model, variant):
        if variant == 'A':
            model.late_fusion = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
            model.fusion_norm = nn.LayerNorm(HIDDEN_DIM)
        if variant in {'B', 'D'}:
            model.cross_t2i = nn.MultiheadAttention(embed_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=ATTN_DROPOUT, batch_first=True)
            model.cross_fusion = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
            model.fusion_norm = nn.LayerNorm(HIDDEN_DIM)
        if variant in {'C', 'C_NO_INCONGRUITY', 'C_NO_GATE'}:
            model.cross_t2i = nn.MultiheadAttention(embed_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=ATTN_DROPOUT, batch_first=True)
            model.cross_i2t = nn.MultiheadAttention(embed_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=ATTN_DROPOUT, batch_first=True)
            model.ca_projection = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
            model.ca_norm = nn.LayerNorm(HIDDEN_DIM)
            if variant in {'C', 'C_NO_GATE'}:
                model.text_incongruity_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
                model.image_incongruity_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
                model.icam_norm = nn.LayerNorm(HIDDEN_DIM)
            if variant == 'C':
                model.gate = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        if model.uses_topic:
            model.topic_projection = nn.Linear(8, TOPIC_PROJ_DIM)
            classifier_input = HIDDEN_DIM + TOPIC_PROJ_DIM
        else:
            classifier_input = HIDDEN_DIM
        model.classifier = nn.Sequential(nn.Linear(classifier_input, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    @classmethod
    def build_from_checkpoint_state(cls, text_spec, vision_spec, tokenizer, state, local_files_only=True):
        """Construct ICAM sequentially while consuming trainable_state to avoid PeakRSS explosion."""
        import gc
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        variant = "C"
        model.tokenizer = tokenizer
        model.variant = variant
        model.uses_topic = True

        model.text_encoder = AutoModel.from_pretrained(text_spec['id'], revision=text_spec['revision'], local_files_only=local_files_only)
        if int(model.text_encoder.config.hidden_size) != HIDDEN_DIM:
            raise RuntimeError('Unexpected IndoBERTweet hidden dimension.')
        set_module_trainable(model.text_encoder.embeddings, False)
        for layer in model.text_encoder.encoder.layer[:2]:
            set_module_trainable(layer, False)
        with torch.no_grad():
            for name, param in model.text_encoder.named_parameters():
                if not param.requires_grad:
                    continue
                full = f'text_encoder.{name}'
                if full not in state:
                    raise KeyError(full)
                tensor = state.pop(full)
                param.copy_(tensor)
                del tensor
        gc.collect()

        model.vision_encoder = AutoModel.from_pretrained(vision_spec['id'], revision=vision_spec['revision'], local_files_only=local_files_only)
        if int(model.vision_encoder.config.hidden_size) != HIDDEN_DIM:
            raise RuntimeError('Unexpected ViT hidden dimension.')
        set_module_trainable(model.vision_encoder.embeddings, False)
        for layer in model.vision_encoder.encoder.layer[:2]:
            set_module_trainable(layer, False)
        with torch.no_grad():
            for name, param in model.vision_encoder.named_parameters():
                if not param.requires_grad:
                    continue
                full = f'vision_encoder.{name}'
                if full not in state:
                    raise KeyError(full)
                tensor = state.pop(full)
                param.copy_(tensor)
                del tensor
        gc.collect()

        cls._init_fusion_heads(model, variant)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.startswith(('text_encoder.', 'vision_encoder.')) or not param.requires_grad:
                    continue
                if name not in state:
                    raise KeyError(name)
                tensor = state.pop(name)
                param.copy_(tensor)
                del tensor
        if state:
            raise KeyError(f'Unexpected extra checkpoint keys: {sorted(state)[:5]}')
        gc.collect()
        return model

    def __init__(self, text_spec, vision_spec, tokenizer, local_files_only=True):
        super().__init__()
        variant = "C"
        self.tokenizer = tokenizer
        if variant not in {'A', 'B', 'C', 'D', 'C_NO_INCONGRUITY', 'C_NO_GATE'}:
            raise ValueError(f'Variant tidak dikenal: {variant}')
        self.variant = variant
        self.uses_topic = variant in {'A', 'B', 'C', 'C_NO_INCONGRUITY', 'C_NO_GATE'}
        self.text_encoder = AutoModel.from_pretrained(text_spec['id'], revision=text_spec['revision'], local_files_only=local_files_only)
        self.vision_encoder = AutoModel.from_pretrained(vision_spec['id'], revision=vision_spec['revision'], local_files_only=local_files_only)
        if int(self.text_encoder.config.hidden_size) != HIDDEN_DIM:
            raise RuntimeError('Unexpected IndoBERTweet hidden dimension.')
        if int(self.vision_encoder.config.hidden_size) != HIDDEN_DIM:
            raise RuntimeError('Unexpected ViT hidden dimension.')
        freeze_controlled_backbones(self.text_encoder, self.vision_encoder)
        self._init_fusion_heads(self, variant)

    def semantic_text_mask(self, input_ids, attention_mask):
        mask = attention_mask.bool().clone()
        for token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
            if token_id is not None:
                mask &= input_ids.ne(int(token_id))
        empty = mask.sum(dim=1).eq(0)
        if empty.any():
            fallback = attention_mask.bool().clone()
            if self.tokenizer.cls_token_id is not None:
                fallback &= input_ids.ne(self.tokenizer.cls_token_id)
            mask[empty] = fallback[empty]
        return mask

    def forward(self, input_ids, attention_mask, pixel_values, topic_features=None, return_diagnostics=False):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        vision_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        text_tokens = text_outputs.last_hidden_state
        vision_tokens = vision_outputs.last_hidden_state
        text_cls = text_tokens[:, 0, :]
        image_cls = vision_tokens[:, 0, :]
        image_patches = vision_tokens[:, 1:, :]
        diagnostics = {}
        if self.variant == 'A':
            representation = self.fusion_norm(self.late_fusion(torch.cat([text_cls, image_cls], dim=-1)))
        elif self.variant in {'B', 'D'}:
            attended_image, attention_weights = self.cross_t2i(query=text_cls.unsqueeze(1), key=image_patches, value=image_patches, need_weights=return_diagnostics, average_attn_weights=False)
            attended_image = attended_image.squeeze(1)
            representation = self.fusion_norm(self.cross_fusion(torch.cat([text_cls, attended_image], dim=-1)))
            if return_diagnostics:
                diagnostics['t2i_attention'] = attention_weights
        else:
            z_t2i, t2i_weights = self.cross_t2i(query=text_cls.unsqueeze(1), key=image_patches, value=image_patches, need_weights=return_diagnostics, average_attn_weights=False)
            z_t2i = z_t2i.squeeze(1)
            z_i2t, i2t_weights = self.cross_i2t(query=image_cls.unsqueeze(1), key=text_tokens, value=text_tokens, key_padding_mask=~attention_mask.bool(), need_weights=return_diagnostics, average_attn_weights=False)
            z_i2t = z_i2t.squeeze(1)
            z_ca = self.ca_norm(self.ca_projection(torch.cat([z_t2i, z_i2t], dim=-1)))
            if self.variant == 'C_NO_INCONGRUITY':
                representation = z_ca
                if return_diagnostics:
                    diagnostics.update({'t2i_attention': t2i_weights, 'i2t_attention': i2t_weights})
            else:
                semantic_mask = self.semantic_text_mask(input_ids, attention_mask)
                text_projected = F.normalize(self.text_incongruity_projection(text_tokens), p=2, dim=-1)
                image_projected = F.normalize(self.image_incongruity_projection(image_patches), p=2, dim=-1)
                similarity = torch.einsum('bth,bph->btp', text_projected, image_projected)
                similarity = similarity.masked_fill(~semantic_mask.unsqueeze(-1), -10000.0)
                max_similarity = similarity.max(dim=1).values
                mismatch = 1.0 - max_similarity
                incongruity_weights = torch.softmax(mismatch, dim=-1)
                z_inc = torch.einsum('bp,bph->bh', incongruity_weights, image_projected)
                if self.variant == 'C':
                    gate_values = torch.sigmoid(self.gate(torch.cat([z_ca, z_inc], dim=-1)))
                    representation = self.icam_norm(gate_values * z_ca + (1.0 - gate_values) * z_inc)
                elif self.variant == 'C_NO_GATE':
                    representation = self.icam_norm(0.5 * z_ca + 0.5 * z_inc)
                if return_diagnostics:
                    diagnostics.update({'t2i_attention': t2i_weights, 'i2t_attention': i2t_weights, 'incongruity_weights': incongruity_weights})
                    if self.variant == 'C':
                        diagnostics['gate_values'] = gate_values
        if self.uses_topic:
            if topic_features is None:
                raise ValueError(f'Variant {self.variant} membutuhkan topic_features.')
            topic_embedding = self.topic_projection(topic_features)
            classifier_input = torch.cat([representation, topic_embedding], dim=-1)
        else:
            if topic_features is not None:
                raise ValueError('Variant D tidak boleh menerima topic_features.')
            classifier_input = representation
        logits = self.classifier(classifier_input)
        if return_diagnostics:
            diagnostics['representation'] = representation
            return (logits, diagnostics)
        return logits
