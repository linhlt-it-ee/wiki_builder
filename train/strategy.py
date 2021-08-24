import torch
import numpy as np
from math import ceil
from sklearn.metrics import pairwise_distances

def mask2idx(mask: torch.BoolTensor) -> np.ndarray:
    return torch.where(mask)[0].cpu().numpy()

class Strategy:
    def __init__(self, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
        self.unlabel_mask = unlabel_mask.cpu()
        self.n_samples = len(self.unlabel_mask)
        self.train_mask = torch.zeros(self.n_samples, dtype=torch.bool)
        self.n_samples_per_round = ceil(sum(self.unlabel_mask).item() / n_rounds)
        self.randomizer = np.random.RandomState(seed=1)

    def _get_mask(self, select_idx) -> torch.BoolTensor:
        self.train_mask[select_idx] = True
        self.unlabel_mask[select_idx] = False
        return self.train_mask

    def random_mask(self):
        train_idx = mask2idx(self.unlabel_mask)
        rand_idx = self.randomizer.choice(train_idx, size=self.n_samples_per_round)
        return self._get_mask(rand_idx)

    def query(self, probs: torch.Tensor = None, logits: torch.Tensor = None):
        pass

class RandomSampling(Strategy):
    def query(self, probs: torch.Tensor = None, logits: torch.Tensor = None):
        return super().random_mask()

class LeastConfidence(Strategy):
    def query(self, probs: torch.Tensor = None, logits: torch.Tensor = None):
        confidence = probs[self.unlabel_mask].max(dim=1).values
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[confidence.sort().indices[:self.n_samples_per_round]]
        return self._get_mask(select_idx)

class EntropySampling(Strategy):
    def query(self, probs: torch.Tensor = None, logits: torch.Tensor = None):
        p = probs[self.unlabel_mask]
        entropy = (p * p.log()).sum(dim=1)
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[entropy.sort().indices[:self.n_samples_per_round]]
        return self._get_mask(select_idx)

class KCenterGreedy(Strategy):
    def query(self, probs: torch.Tensor = None, logits: torch.Tensor = None):
        logits = logits.numpy()
        dist = pairwise_distances(logits, logits, metric="l2")
        center_dist = dist[self.unlabel_mask][:, self.train_mask]
        # iteratively pick an outliner (farthest from its center)
        for _ in range(self.n_samples_per_round):
            nearest_center_dist = center_dist.min(axis=1)
            idx = nearest_center_dist.argmax()
            select_idx = torch.arange(self.n_samples)[self.unlabel_mask][idx]
            self.unlabel_mask[select_idx] = False
            self.train_mask[select_idx] = True
            # remove `idx` as an unlabeled sample and append it as a new center
            center_dist = np.delete(center_dist, idx, axis=0)
            center_dist = np.append(center_dist, dist[self.unlabel_mask, idx][:, None], axis=1)

        return self.train_mask

def prepare_strategy(strategy_name: str, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
    if strategy_name == "random":
        return RandomSampling(unlabel_mask, n_rounds)
    elif strategy_name == "lc":
        return LeastConfidence(unlabel_mask, n_rounds)
    elif strategy_name == "entropy":
        return EntropySampling(unlabel_mask, n_rounds)
    elif strategy_name == "kcenter":
        return KCenterGreedy(unlabel_mask, n_rounds)
    else:
        raise NotImplementedError
