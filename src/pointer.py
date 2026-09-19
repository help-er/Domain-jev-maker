"""The pointer readout.

Each option is scored from its own hidden state against the decision position:

    z_i = <W_q h_dec, W_k h_opt_i> / sqrt(d)

Two properties follow, and both are load-bearing for the recipe:

  * **Options are scored independently.** Nothing the model learns is tied to
    how many options were present, so distractors can be subsampled during
    training and the full set used at inference.
  * **No label token is involved.** Options are rendered as bare criteria text,
    so the option set is not limited by single-token labels and K can be large.
"""
import math

import torch
import torch.nn as nn


class PointerHead(nn.Module):
    """Low-rank bilinear scorer.

    d_model -> 256 keeps it near 0.8M parameters on a 1536-dim backbone, which
    is trainable on a few thousand examples; a full d x d bilinear form would
    be 2.4M and mostly unconstrained.
    """

    def __init__(self, d_model, d_proj=256):
        super().__init__()
        self.q = nn.Linear(d_model, d_proj, bias=False)
        self.k = nn.Linear(d_model, d_proj, bias=False)
        self.scale = 1.0 / math.sqrt(d_proj)
        nn.init.normal_(self.q.weight, std=0.02)
        nn.init.normal_(self.k.weight, std=0.02)

    def forward(self, h_dec, h_opt):
        # h_dec (B, D) ; h_opt (B, K, D)
        qv = self.q(h_dec).unsqueeze(1)          # (B, 1, P)
        kv = self.k(h_opt)                       # (B, K, P)
        return (qv * kv).sum(-1) * self.scale    # (B, K)


def logits_for(model, head, batch, device, dtype):
    """One forward pass over the packed batch, then the pointer scores.

    The 4D block mask is materialised as an additive float mask because that is
    what the backbone expects; `mask4d` is the boolean "may attend" matrix.
    """
    m = torch.zeros(batch["mask4d"].shape, dtype=dtype)
    m.masked_fill_(~batch["mask4d"], torch.finfo(dtype).min)
    out = model(input_ids=batch["input_ids"].to(device),
                position_ids=batch["position_ids"].to(device),
                attention_mask=m.to(device),
                output_hidden_states=True)
    h = out.hidden_states[-1]
    b = torch.arange(h.size(0), device=device)
    h_dec = h[b, batch["dec"].to(device)]
    h_opt = h[b.unsqueeze(1), batch["opt"].to(device)]
    z = head(h_dec.float(), h_opt.float())
    return z.masked_fill(~batch["valid"].to(device), float("-inf"))
