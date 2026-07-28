import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from tqdm import tqdm

from .base import BaseTask


class ClassificationTask(BaseTask):
    """Sequence-level (whole-window) classification.

    The model produces a single vector of K logits per input sequence
    (see MedTsLLM.predict / forward), trained with cross-entropy and
    evaluated with accuracy / macro F1 / macro precision / macro recall,
    matching the PTB-XL results table in the paper.
    """

    def __init__(self, run_id, config, newrun=True):
        self.task = "classification"
        super(ClassificationTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
                    pred = self.model(inputs)          # [bs, n_classes] (raw logits in train mode)
                    labels = inputs["labels"]          # [bs] long
                    loss = self.loss_fn(pred, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_step(loss.item())

            val_scores = self.val()
            test_scores = self.test()
            self.log_epoch(val_scores)
            self.scheduler.step()
            print(f"[epoch {epoch + 1}] val:  {val_scores}")
            print(f"[epoch {epoch + 1}] test: {test_scores}")
        self.model.eval()

    def val(self):
        preds, targets = self.predict(self.val_dataloader)
        scores = self.score(preds, targets)
        scores = {f"val/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def test(self):
        preds, targets = self.predict(self.test_dataloader)
        scores = self.score(preds, targets)
        scores = {f"test/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def predict(self, dataloader):
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs in tqdm(dataloader, total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                pred = self.model(inputs)              # [bs, n_classes] (probabilities in eval mode)
                all_preds.append(pred.float().cpu())
                all_targets.append(inputs["labels"].cpu())

        preds = torch.cat(all_preds, dim=0)            # [N, n_classes]
        targets = torch.cat(all_targets, dim=0)        # [N]

        return preds, targets

    def build_loss(self):
        match self.config.training.loss:
            case "ce" | "cross_entropy" | "auto":
                weight = None
                if self.config.training.get("class_weights", False):
                    weight = self.train_dataset.class_weights.to(self.device)
                self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            case _:
                raise ValueError(f"Invalid loss function selection: {self.config.training.loss}")
        return self.loss_fn

    def score(self, pred_scores, target):
        pred = pred_scores.argmax(dim=1).int().numpy()
        target = target.int().numpy()
        avg_mode = "binary" if pred_scores.size(1) == 2 else "macro"
        return {
            "accuracy": accuracy_score(target, pred),
            "f1": f1_score(target, pred, average=avg_mode, zero_division=0),
            "precision": precision_score(target, pred, average=avg_mode, zero_division=0),
            "recall": recall_score(target, pred, average=avg_mode, zero_division=0),
        }
