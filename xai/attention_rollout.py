from xai.reference import capture_vision_tokens_and_rollout

def attention_rollout(model, pixel_values):
    _, rollout = capture_vision_tokens_and_rollout(model, pixel_values)
    return rollout[0]
