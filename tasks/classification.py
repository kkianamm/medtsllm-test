"""
Whole-sequence classification task for MedTsLLM + TEST.

Two stages, selected by `config.training.stage`:

  stage = "pretrain"  (Stage A)
      Unsupervised text-prototype-aligned contrastive pretraining of the
      TS-encoding stack (patch embedding + reprogramming + projection).
      The LLM is not used, so this is cheap.  Labels are ignored.
      eval metric: align_loss (min).

  stage = "classify"  (Stage B)
      Supervised classification with the frozen (or LoRA) LLM.  Loss is
      cross-entropy plus an optional alignment regulariser
      (config.training.lambda_align).  Loads Stage-A weights from
      config.training.pretrained_path if given.
      eval metric: accuracy / f1_macro / auroc (max).
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .base import BaseTask


class ClassificationTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "classification"
        self.stage = config.training.get("stage", "classify")
        self.lambda_align = float(config.training.get("lambda_align", 0.0))
        super(ClassificationTask, self).__init__(run_id, config, newrun)

    # ------------------------------------------------------------------ #
    #  weight loading (Stage-B loads Stage-A checkpoint)
    # ------------------------------------------------------------------ #
    def load_pretrained(self):
        self.finetuning = False
        self.loaded_params = []
        path = self.config.training.get("pretrained_path", None)
        if path in (None, "", "none"):
            return
        state = torch.load(path, map_location="cpu")
        state = state["model"] if "model" in state else state
        self.loaded_params = self.model.load_pretrained(state)
        print(f"Loaded {len(self.loaded_params)} pretrained tensors from {path}")

    # ------------------------------------------------------------------ #
    #  training
    # ------------------------------------------------------------------ #
    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs} [{self.stage}]")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
                    if self.stage == "pretrain":
                        loss, _ = self.model.alignment_loss(inputs["x_enc"])
                    else:
                        logits = self.model.classification_logits(inputs)
                        loss = self.loss_fn(logits, self._targets(inputs))
                        if self.lambda_align > 0:
                            aux, _ = self.model.alignment_loss(inputs["x_enc"])
                            loss = loss + self.lambda_align * aux

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.log_step(loss.item())

            val_scores = self.val()
            self.log_epoch(val_scores)
            self.scheduler.step()

        self.model.eval()

    def _targets(self, inputs):
        y = inputs["label"] if "label" in inputs else inputs["labels"]
        return y.long().view(-1)

    # ------------------------------------------------------------------ #
    #  eval
    # ------------------------------------------------------------------ #
    def val(self):
        if self.stage == "pretrain":
            scores = {"val/align_loss": self._mean_align(self.val_dataloader)}
            self.log_scores(scores)
            return scores
        preds, targets = self.predict(self.val_dataloader)
        scores = {f"val/{k}": v for k, v in self.score(preds, targets).items()}
        self.log_scores(scores)
        return scores

    def test(self):
        if self.stage == "pretrain":
            scores = {"test/align_loss": self._mean_align(self.test_dataloader)}
            self.log_scores(scores)
            return scores
        preds, targets = self.predict(self.test_dataloader)
        scores = {f"test/{k}": v for k, v in self.score(preds, targets).items()}
        self.log_scores(scores)
        return scores

    @torch.no_grad()
    def _mean_align(self, dataloader):
        self.model.eval()
        tot, n = 0.0, 0
        for inputs in tqdm(dataloader):
            inputs = self.prepare_batch(inputs)
            loss, _ = self.model.alignment_loss(inputs["x_enc"])
            tot += float(loss.detach()) * inputs["x_enc"].size(0)
            n += inputs["x_enc"].size(0)
        return tot / max(n, 1)

    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()
        all_probs, all_targets = [], []
        for inputs in tqdm(dataloader):
            inputs = self.prepare_batch(inputs)
            probs = self.model(inputs)                       # eval -> softmax probs
            all_probs.append(probs.float().cpu())
            all_targets.append(self._targets(inputs).cpu())
        return torch.cat(all_probs), torch.cat(all_targets)

    def score(self, probs, targets):
        y = targets.numpy()
        p = probs.numpy()
        pred = p.argmax(axis=1)
        n_classes = p.shape[1]
        out = {
            "accuracy": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        }
        try:
            if n_classes == 2:
                out["auroc"] = roc_auc_score(y, p[:, 1])
            else:
                out["auroc"] = roc_auc_score(y, p, multi_class="ovr", average="macro")
        except ValueError:
            out["auroc"] = float("nan")
        return out

    # ------------------------------------------------------------------ #
    def build_loss(self):
        if self.stage == "pretrain":
            self.loss_fn = nn.Identity()      # loss computed inside model
            return self.loss_fn
        weight = None
        cw = self.config.training.get("class_weights", None)
        if cw:
            weight = torch.tensor(cw, dtype=torch.float32, device=self.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        return self.loss_fn
