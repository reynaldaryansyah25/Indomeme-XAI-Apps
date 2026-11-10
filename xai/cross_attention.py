def cross_attention(diagnostics):
    return diagnostics['t2i_attention'].squeeze(2).float().mean(dim=1)[0]
