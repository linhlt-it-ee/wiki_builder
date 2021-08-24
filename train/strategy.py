import torch
import numpy as np
from math import ceil

def mask2idx(mask: torch.BoolTensor) -> np.ndarray:
    return torch.where(mask)[0].cpu().numpy()

class Strategy:
    def __init__(self, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
        self.unlabel_mask = unlabel_mask
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

    def query(self, probs: torch.Tensor = None):
        pass

class RandomSampling(Strategy):
    def query(self, probs: torch.Tensor = None):
        return super().random_mask()

class LeastConfidence(Strategy):
    def query(self, probs: torch.Tensor = None):
        confidence = probs[self.unlabel_mask].max(dim=1).values
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[confidence.sort().indices[:self.n_samples_per_round]]
        return self._get_mask(select_idx)

class EntropySampling(Strategy):
    def query(self, probs: torch.Tensor = None):
        p = probs[self.unlabel_mask]
        entropy = (p * p.log()).sum(dim=1)
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[entropy.sort().indices[:self.n_samples_per_round]]
        return self._get_mask(select_idx)

def prepare_strategy(strategy_name: str, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
    if strategy_name == "random":
        return RandomSampling(unlabel_mask, n_rounds)
    elif strategy_name == "lc":
        return LeastConfidence(unlabel_mask, n_rounds)
    elif strategy_name == "entropy":
        return EntropySampling(unlabel_mask, n_rounds)
    else:
        raise NotImplementedError
