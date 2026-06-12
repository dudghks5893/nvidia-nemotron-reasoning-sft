"""Pattern/sample weighted Trainer utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Trainer


class PatternWeightedTrainer(Trainer):
    """Trainer that applies sample-level loss_weight after assistant-token averaging.

    Expected batch key:
        loss_weight: FloatTensor of shape [batch]

    Labels must already use -100 for masked prompt tokens.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weight = inputs.pop("loss_weight", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        valid_mask = shift_labels.ne(-100)
        token_count_per_sample = valid_mask.sum(dim=1).clamp_min(1)
        sample_loss = (token_loss * valid_mask).sum(dim=1) / token_count_per_sample

        if loss_weight is not None:
            loss_weight = loss_weight.to(device=sample_loss.device, dtype=sample_loss.dtype)
            loss = (sample_loss * loss_weight).sum() / loss_weight.sum().clamp_min(1e-8)
        else:
            loss = sample_loss.mean()

        return (loss, outputs) if return_outputs else loss


class DataCollatorWithLossWeight:
    """Wrap an existing collator and preserve loss_weight as a batch tensor."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        loss_weights = []
        cleaned = []
        for feature in features:
            feature = dict(feature)
            loss_weights.append(float(feature.pop("loss_weight", 1.0)))
            cleaned.append(feature)
        batch = self.base_collator(cleaned)
        batch["loss_weight"] = torch.tensor(loss_weights, dtype=torch.float32)
        return batch
