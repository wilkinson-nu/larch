import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math

class SupervisedHead(nn.Module):
    def __init__(self, encoder_dim, classifier_config):
        super().__init__()
        self.classifier_config = classifier_config
        self.heads = nn.ModuleDict({
            name: nn.Linear(encoder_dim, cfg['n_classes'])
            for name, cfg in classifier_config.items()
        })

    def forward(self, features):
        return {name: head(features) for name, head in self.heads.items()}

    
def supervised_loss(outputs, labels, classifier_config):
    total_loss = 0.0
    loss_dict = {}
    for name, cfg in classifier_config.items():
        loss = F.cross_entropy(outputs[name], labels[name].long())
        weighted_loss = cfg['weight'] * loss
        total_loss += weighted_loss
        loss_dict[name] = loss.detach()
    return total_loss, loss_dict


class ClassificationMetrics:
    """
    Computes, for every particle/count target:

      - MAE of the predicted multiplicity
      - Binary F1 for zero versus nonzero

    For nonzero F1, support from all multiplicity classes greater than zero
    is combined before deciding whether the prediction is nonzero.
    """

    def __init__(self, classifier_config, device):
        self.classifier_config = classifier_config
        self.device = device
        self.reset()

    def reset(self):
        self.total = {
            name: torch.zeros((), device=self.device)
            for name in self.classifier_config
        }
        self.abs_error = {
            name: torch.zeros((), device=self.device)
            for name in self.classifier_config
        }
        self.nonzero_tp = {
            name: torch.zeros((), device=self.device)
            for name in self.classifier_config
        }
        self.nonzero_fp = {
            name: torch.zeros((), device=self.device)
            for name in self.classifier_config
        }
        self.nonzero_fn = {
            name: torch.zeros((), device=self.device)
            for name in self.classifier_config
        }

    @torch.no_grad()
    def update(
        self,
        outputs,
        labels,
        *,
        outputs_are_logits=False,
    ):
        """
        Parameters
        ----------
        outputs:
            Dictionary mapping label names to tensors of shape [N, C].

            For kNN, these are nonnegative class votes.
            For a linear classifier, these are usually logits.

        labels:
            Dictionary mapping label names to integer targets of shape [N].

        outputs_are_logits:
            Set True for ordinary classifier logits. Leave False for kNN
            votes or probabilities.
        """
        for name, cfg in self.classifier_config.items():
            scores = outputs[name]
            targets = labels[name].long()

            ## Multiplicity MAE
            predictions = scores.argmax(dim=-1)

            self.abs_error[name] += (predictions.float() - targets.float()).abs().sum()
            self.total[name] += targets.numel()

            ## Binary zero-versus-nonzero F1
            ## Convert raw logits to probabilities
            if outputs_are_logits:
                presence_scores = scores.softmax(dim=-1)
            else:
                ## kNN votes or already-normalized probabilities.
                presence_scores = scores

            zero_score = presence_scores[:, 0]
            nonzero_score = presence_scores[:, 1:].sum(dim=-1)

            ## Treat ties as zero
            predicted_nonzero = nonzero_score > zero_score
            target_nonzero = targets > 0

            self.nonzero_tp[name] += (predicted_nonzero & target_nonzero).sum()
            self.nonzero_fp[name] += (predicted_nonzero & ~target_nonzero).sum()
            self.nonzero_fn[name] += (~predicted_nonzero & target_nonzero).sum()

    def reduce(self):

        for name in self.classifier_config:
            packed = torch.stack([
                self.total[name],
                self.abs_error[name],
                self.nonzero_tp[name],
                self.nonzero_fp[name],
                self.nonzero_fn[name],
            ])

            dist.all_reduce(
                packed,
                op=dist.ReduceOp.SUM,
            )

            self.total[name] = packed[0]
            self.abs_error[name] = packed[1]
            self.nonzero_tp[name] = packed[2]
            self.nonzero_fp[name] = packed[3]
            self.nonzero_fn[name] = packed[4]

    def compute(self):
        results = {}

        for name in self.classifier_config:
            total = self.total[name].item()
            abs_error = self.abs_error[name].item()

            tp = self.nonzero_tp[name].item()
            fp = self.nonzero_fp[name].item()
            fn = self.nonzero_fn[name].item()

            mae = (
                abs_error / total
                if total > 0
                else float("nan")
            )

            f1_denominator = 2 * tp + fp + fn

            f1_nonzero = (
                2 * tp / f1_denominator
                if f1_denominator > 0
                else float("nan")
            )

            results[name] = {
                "mae": mae,
                "f1_nonzero": f1_nonzero,
            }

        return results
