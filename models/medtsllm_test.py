"""
MedTsLLMTest: MedTsLLM + TEST text-prototype alignment, for time-series
classification.

It subclasses the original MedTsLLM so all of the reprogramming / multivariate
/ prompting machinery is reused untouched.  On top of it we add:

  * text prototypes selected from the LLM's own word-embedding matrix
    (TEST, ICLR 2024), registered as a buffer;
  * a TestAlignmentLoss (instance / feature / text-prototype contrast) applied
    to the reprogrammed TS token embeddings -- used both as the Stage-A
    pretraining objective and as a Stage-B auxiliary regulariser;
  * learnable soft-prompt tokens prepended to the LLM input (TEST);
  * a pooling classification head for whole-sequence classification.

No edits to models/medtsllm.py are required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .medtsllm import MedTsLLM
from losses.test_alignment import TestAlignmentLoss, select_text_prototypes


class MedTsLLMTest(MedTsLLM):

    # add 'classification' so BaseTask.build_model's assertion passes
    supported_tasks = MedTsLLM.supported_tasks + ["classification"]

    def __init__(self, config, dataset):
        # The parent __init__ raises on task='classification' (it only knows
        # segmentation/forecasting/...).  Build it under a segmentation-shaped
        # task so all layers are created identically, then restore.
        real_task = config.task
        is_classification = (real_task == "classification")
        if is_classification:
            config.task = "semantic_segmentation"
        try:
            super().__init__(config, dataset)
        finally:
            config.task = real_task
        self.task = real_task

        # ------- TEST hyper-parameters (read from models.<name>.test) -------- #
        tcfg = self.model_config.get("test", None)
        # sensible defaults if the block is missing
        defaults = dict(
            n_prototypes=64, prototype_mode="representative",
            prototype_words="", soft_prompt_len=8,
            w_instance=1.0, w_feature=0.5, w_text=1.0,
            proj_dim=128, temperature=0.1, text_mode="soft",
            pooling="mean",
        )
        def _get(k):
            if tcfg is None:
                return defaults[k]
            return tcfg.get(k, defaults[k])

        self.n_prototypes = int(_get("n_prototypes"))
        self.prototype_mode = _get("prototype_mode")
        self.soft_prompt_len = int(_get("soft_prompt_len"))
        self.pooling = _get("pooling")

        # ------- text prototypes (buffer, from the frozen LLM vocab) --------- #
        words = _get("prototype_words")
        words = [w.strip() for w in words.split(",") if w.strip()] if isinstance(words, str) else words
        tokenizer = getattr(self, "tokenizer", None)
        protos = select_text_prototypes(
            self.word_embeddings,
            mode=self.prototype_mode,
            n_prototypes=self.n_prototypes,
            words=words if self.prototype_mode == "provided" else None,
            tokenizer=tokenizer,
        )
        self.register_buffer("text_prototypes", protos.to(torch.float32))

        # ------- alignment loss module -------------------------------------- #
        self.alignment = TestAlignmentLoss(
            w_instance=float(_get("w_instance")),
            w_feature=float(_get("w_feature")),
            w_text=float(_get("w_text")),
            temperature=float(_get("temperature")),
            text_mode=_get("text_mode"),
            proj_dim=int(_get("proj_dim")) if _get("proj_dim") else None,
            d_model=self.d_llm,
        )

        # ------- soft prompt ------------------------------------------------- #
        if self.soft_prompt_len > 0:
            self.soft_prompt = nn.Parameter(
                0.02 * torch.randn(self.soft_prompt_len, self.d_llm)
            )
        else:
            self.register_parameter("soft_prompt", None)

        # ------- classification head ---------------------------------------- #
        if is_classification:
            self.class_head = nn.Sequential(
                nn.LayerNorm(self.d_ff),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.n_classes),
            )
            # a clearer task description than the segmentation default
            self.task_description = (
                f"Classify the following {self.seq_len}-step signal into one of "
                f"{self.n_classes} categories."
            )
        else:
            self.class_head = None

    # --------------------------------------------------------------------- #
    #  Per-feature reprogrammed tokens  ->  [B, F, P, D]   (for alignment)
    # --------------------------------------------------------------------- #
    def encode_tokens(self, x_enc):
        assert self.covariate_mode != "concat", (
            "Stage-A / alignment loss needs a per-feature token layout; "
            "covariate_mode='concat' merges features before reprogramming. "
            "Use 'interleave', 'independent', 'add', 'weighted-average', or "
            "'univariate' for the TEST alignment path."
        )
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)
        bs, seq_len, n_features = x_enc.size()

        x = self.normalize_layers(x_enc, "norm")
        x = x.permute(0, 2, 1).contiguous()                 # [B, F, L]
        enc, _ = self.patch_embedding(x)                    # [B*F, P, d_patch]
        P = enc.size(1)

        source = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc = self.reprogramming_layer(enc, source, source)  # [B*F, P, d_llm]
        enc = enc.reshape(bs, n_features, P, self.d_llm)     # [B, F, P, D]
        return enc

    def alignment_loss(self, x_enc):
        tokens = self.encode_tokens(x_enc)
        protos = self.text_prototypes.to(tokens.dtype)
        return self.alignment(tokens, protos)

    # --------------------------------------------------------------------- #
    #  LLM forward -> per-patch hidden states downsampled to d_ff
    #  (mirrors MedTsLLM.predict up to just before output_projection, plus
    #   an optional soft prompt)
    # --------------------------------------------------------------------- #
    def _llm_hidden(self, inputs):
        x_enc = inputs["x_enc"]
        bs, seq_len, n_features = x_enc.size()
        if self.device is None:
            self.device = x_enc.device

        prompts = self.build_prompt(inputs)
        if len(prompts[0]) > 0:
            prompt_enc = [[self.encode_part(p) for p in prompt] for prompt in prompts]
            prompt_enc = [torch.cat(enc, dim=1) for enc in prompt_enc]
            max_len = max(e.size(1) for e in prompt_enc)
            prompt_enc = [self.pad_sequence(e, max_len) for e in prompt_enc]
            prompt_enc = torch.cat(prompt_enc, dim=0)
        else:
            prompt_enc = torch.zeros((bs, 0, self.d_llm), device=x_enc.device, dtype=x_enc.dtype)

        x_ts = self.encode_ts(x_enc)                        # respects covariate_mode

        if self.covariate_mode in ("independent", "merge-end"):
            prompt_enc = prompt_enc.repeat_interleave(n_features, dim=0)

        if self.soft_prompt is not None:
            sp = self.soft_prompt.unsqueeze(0).expand(x_ts.size(0), -1, -1).to(x_ts.dtype)
            prompt_enc = torch.cat([sp, prompt_enc], dim=1)

        if self.llm.config.is_encoder_decoder:
            dec_out = self.llm(inputs_embeds=prompt_enc, decoder_inputs_embeds=x_ts).last_hidden_state
        else:
            enc = torch.cat([prompt_enc, x_ts], dim=1)
            dec_out = self.llm(inputs_embeds=enc).last_hidden_state
        dec_out = dec_out.to(x_ts.dtype)

        dec_out = dec_out[:, -self.n_patches:, :]           # [., n_patches, d_llm]
        match self.embedding_downsample_mode:
            case "truncate":
                dec_out = dec_out[:, :, :self.d_ff]
            case "linear":
                dec_out = self.embedding_downsample_layer(dec_out)
            case "average":
                dec_out = dec_out.reshape(dec_out.size(0), self.n_patches, self.d_ff, -1).mean(dim=-1)
            case _:
                raise ValueError(f"Unknown embedding downsample mode {self.embedding_downsample_mode}")
        return dec_out                                      # [B or B*F, n_patches, d_ff]

    def classification_logits(self, inputs):
        assert self.class_head is not None, "model was not built for classification"
        dec_out = self._llm_hidden(inputs)                  # [B*, n_patches, d_ff]

        if self.pooling == "max":
            pooled = dec_out.max(dim=1).values
        else:
            pooled = dec_out.mean(dim=1)                     # [B*, d_ff]

        bs = inputs["x_enc"].size(0)
        if pooled.size(0) != bs:                            # independent / merge-end
            n_feat = pooled.size(0) // bs
            pooled = pooled.reshape(bs, n_feat, self.d_ff).mean(dim=1)

        return self.class_head(pooled)                      # [B, n_classes]

    # --------------------------------------------------------------------- #
    def forward(self, inputs):
        if self.task == "classification":
            logits = self.classification_logits(inputs)
            if not self.training:
                if self.n_classes == 2:
                    logits = F.softmax(logits, dim=-1)
                else:
                    logits = F.softmax(logits, dim=-1)
            return logits
        return super().forward(inputs)
