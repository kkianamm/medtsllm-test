"""
TEST-style text-prototype-aligned representation learning for MedTsLLM.

Reference: Sun et al., "TEST: Text Prototype Aligned Embedding to Activate
LLM's Ability for Time Series", ICLR 2024 (https://arxiv.org/abs/2308.08241).

The original TEST trains a CausalCNN encoder with three contrastive signals
(instance-wise, feature-wise, text-prototype-aligned) so that TS token
embeddings land inside the LLM's word-embedding space.  MedTsLLM already
produces TS token embeddings (the output of its reprogramming layer), so here
we re-express the three TEST signals directly on those batched
[B, F, P, D] embeddings instead of on a separate CNN encoder.

Nothing in this file needs the LLM's forward pass -- it only touches the
frozen word-embedding matrix (for prototype selection) and the reprogrammed
TS embeddings.  That makes Stage-A pretraining very cheap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Text prototype selection
# --------------------------------------------------------------------------- #
def _kmeans(x, k, n_iter=50, seed=0):
    """Plain Lloyd k-means on rows of x -> [k, D] centroids (TEST's
    `select_representative`, vectorised and torch-native)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.size(0), generator=g, device=x.device)[:k]
    centroids = x[idx].clone()
    for _ in range(n_iter):
        d = torch.cdist(x, centroids)               # [N, k]
        assign = d.argmin(dim=1)                     # [N]
        new = centroids.clone()
        for j in range(k):
            sel = x[assign == j]
            if sel.numel() > 0:
                new[j] = sel.mean(dim=0)
        if torch.allclose(new, centroids, atol=1e-6):
            centroids = new
            break
        centroids = new
    return centroids


@torch.no_grad()
def select_text_prototypes(
    word_embeddings,
    mode="representative",
    n_prototypes=64,
    words=None,
    tokenizer=None,
    vocab_subsample=20000,
    seed=0,
):
    """
    Pick `n_prototypes` anchor vectors from the LLM's word-embedding matrix.

    word_embeddings : [vocab, D] tensor (MedTsLLM exposes this as
                      `model.word_embeddings`).
    mode            : 'representative' (k-means over vocab, TEST default),
                      'random', or 'provided' (clinical word list).
    words           : list[str], required for mode='provided'.
    tokenizer       : HF tokenizer, required for mode='provided'.

    Returns [n_prototypes, D] on CPU (register it as a buffer on the model).
    """
    we = word_embeddings.detach().float().cpu()

    if mode == "provided":
        assert words is not None and tokenizer is not None, \
            "mode='provided' needs `words` and `tokenizer`"
        ids = []
        for w in words:
            toks = tokenizer(w, add_special_tokens=False).input_ids
            ids.extend(toks)
        ids = sorted(set(int(i) for i in ids if 0 <= int(i) < we.size(0)))
        protos = we[ids]
        # pad up to n_prototypes with representative centroids if too few
        if protos.size(0) < n_prototypes:
            extra = _kmeans(we[torch.randperm(we.size(0))[:vocab_subsample]],
                            n_prototypes - protos.size(0), seed=seed)
            protos = torch.cat([protos, extra], dim=0)
        return protos[:n_prototypes].contiguous()

    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        ids = torch.randperm(we.size(0), generator=g)[:n_prototypes]
        return we[ids].contiguous()

    # 'representative' -- k-means over a random vocab subsample for speed
    sub = we
    if we.size(0) > vocab_subsample:
        g = torch.Generator().manual_seed(seed)
        sub = we[torch.randperm(we.size(0), generator=g)[:vocab_subsample]]
    return _kmeans(sub, n_prototypes, seed=seed).contiguous()


# --------------------------------------------------------------------------- #
#  Losses
# --------------------------------------------------------------------------- #
def info_nce(anchor, positive, temperature=0.1):
    """Standard InfoNCE with in-batch negatives.
    anchor / positive : [B, D].  positives are on the diagonal."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = a @ p.t() / temperature            # [B, B]
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


class TestAlignmentLoss(nn.Module):
    """
    Combined TEST objective on reprogrammed TS token embeddings.

    forward(tokens, prototypes) where
        tokens     : [B, F, P, D]   (B windows, F covariates, P patches, D=d_llm)
        prototypes : [K, D]         (text prototypes from the LLM vocab)

    Returns (total_loss, parts_dict).
    """

    def __init__(
        self,
        w_instance=1.0,
        w_feature=0.5,
        w_text=1.0,
        temperature=0.1,
        text_mode="soft",          # 'soft' (TEST logsigmoid) or 'ce'
        proj_dim=None,             # optional projection head dim
        d_model=None,
    ):
        super().__init__()
        self.w_instance = w_instance
        self.w_feature = w_feature
        self.w_text = w_text
        self.temperature = temperature
        self.text_mode = text_mode

        self.proj = None
        if proj_dim is not None and d_model is not None:
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, proj_dim),
            )

    def _project(self, x):
        return self.proj(x) if self.proj is not None else x

    # -- instance-wise: two temporal crops of the same window are positives -- #
    def instance_loss(self, tokens):
        B, Fn, P, D = tokens.shape
        if P < 2:
            return tokens.new_zeros(())
        half = P // 2
        v1 = tokens[:, :, :half, :].mean(dim=(1, 2))    # [B, D]
        v2 = tokens[:, :, half:, :].mean(dim=(1, 2))    # [B, D]
        v1, v2 = self._project(v1), self._project(v2)
        return 0.5 * (info_nce(v1, v2, self.temperature)
                      + info_nce(v2, v1, self.temperature))

    # -- feature-wise: covariates within a window should be distinguishable -- #
    def feature_loss(self, tokens):
        B, Fn, P, D = tokens.shape
        if Fn < 2:
            return tokens.new_zeros(())
        feat = tokens.mean(dim=2)                        # [B, F, D]
        feat = self._project(feat)                       # [B, F, D'] (D' may differ)
        feat = F.normalize(feat, dim=-1)
        # For each window, treat feature index as the class: pull the same
        # feature across windows together, push different features apart.
        # anchors: [B*F, D'], labels: feature id
        a = feat.reshape(B * Fn, feat.size(-1))
        labels = torch.arange(Fn, device=tokens.device).repeat(B)
        logits = a @ a.t() / self.temperature           # [B*F, B*F]
        # mask self-similarity
        eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, float("-inf"))
        pos_mask = (labels[:, None] == labels[None, :]) & (~eye)
        # supervised-contrastive style: -log( sum exp(pos) / sum exp(all) )
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        # use where (not multiply) so the -inf diagonal never hits 0*-inf = nan
        pos_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
        denom = pos_mask.sum(dim=1).clamp(min=1)
        loss = -pos_log_prob.sum(dim=1) / denom
        # rows with no positives (denom==1 but none true) contribute 0
        has_pos = pos_mask.any(dim=1)
        if has_pos.any():
            return loss[has_pos].mean()
        return tokens.new_zeros(())

    # -- text-prototype-aligned: pull each TS token to its text prototype ---- #
    def text_loss(self, tokens, prototypes):
        B, Fn, P, D = tokens.shape
        z = tokens.reshape(-1, D)                        # [N, D]
        z = self._project(z)
        proto = self._project(prototypes)               # [K, D]
        zc = F.normalize(z, dim=-1)
        pc = F.normalize(proto, dim=-1)
        sim = zc @ pc.t()                               # [N, K]

        if self.text_mode == "ce":
            # confident assignment to nearest prototype (avoids collapse via
            # entropy over the batch's prototype usage)
            target = sim.argmax(dim=1)
            return F.cross_entropy(sim / self.temperature, target)

        # 'soft' (TEST): build a soft prototype per token, then logsigmoid pull
        attn = F.softmax(sim / self.temperature, dim=1)  # [N, K]
        soft_proto = attn @ pc                           # [N, D]
        soft_proto = F.normalize(soft_proto, dim=-1)
        dot = (zc * soft_proto).sum(dim=-1)              # [N]
        return -F.logsigmoid(dot).mean()

    def forward(self, tokens, prototypes):
        li = self.instance_loss(tokens) if self.w_instance > 0 else tokens.new_zeros(())
        lf = self.feature_loss(tokens) if self.w_feature > 0 else tokens.new_zeros(())
        lt = self.text_loss(tokens, prototypes) if self.w_text > 0 else tokens.new_zeros(())
        total = self.w_instance * li + self.w_feature * lf + self.w_text * lt
        return total, {"instance": li.detach(), "feature": lf.detach(), "text": lt.detach()}
