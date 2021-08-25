import torch
import numpy as np
from math import ceil
from sklearn.metrics import pairwise_distances

class Strategy:
    def __init__(self, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
        self.unlabel_mask = unlabel_mask
        self.n_samples = len(self.unlabel_mask)
        self.train_mask = torch.zeros(self.n_samples, dtype=torch.bool)
        self.n_samples_per_round = ceil(sum(self.unlabel_mask).item() / n_rounds)
        self.randomizer = np.random.RandomState(seed=1)

    def update(self, select_idx):
        self.train_mask[select_idx] = True
        self.unlabel_mask[select_idx] = False

    def random_mask(self) -> torch.BoolTensor:
        train_idx = torch.where(self.unlabel_mask)[0].numpy()
        rand_idx = self.randomizer.choice(train_idx, size=self.n_samples_per_round)
        self.update(rand_idx)
        return self.train_mask

    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        pass

class RandomSampling(Strategy):
    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        return super().random_mask()

class LeastConfidence(Strategy):
    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        confidence = probs[self.unlabel_mask].max(dim=1).values
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[confidence.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class EntropySampling(Strategy):
    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        p = probs[self.unlabel_mask]
        neg_entropy = (p * p.log()).sum(dim=1)
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[neg_entropy.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class MarginSampling(Strategy):
    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        top2_p = probs[self.unlabel_mask].sort(dim=1).values[:, -2:]
        margin = top2_p[:, 1] - top2_p[:, 0]
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[margin.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class MarginSampling2(Strategy):
    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        margin = torch.abs(probs[self.unlabel_mask] - 0.5).min(dim=1).values
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[margin.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class KCenterGreedy(Strategy):
    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        features = features.numpy()
        dist = pairwise_distances(features, features, metric="l2")
        center_dist = dist[self.unlabel_mask][:, self.train_mask]
        # iteratively pick an outliner (farthest from its center)
        for _ in range(self.n_samples_per_round):
            nearest_center_dist = center_dist.min(axis=1)
            idx = nearest_center_dist.argmax()
            select_idx = torch.arange(self.n_samples)[self.unlabel_mask][idx]
            self.update(select_idx)
            # remove `idx` as an unlabeled sample and append it as a new center
            center_dist = np.delete(center_dist, idx, axis=0)
            center_dist = np.append(center_dist, dist[self.unlabel_mask, idx][:, None], axis=1)
        return self.train_mask

def prepare_strategy(strategy_name: str, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
    if strategy_name == "random":
        return RandomSampling(unlabel_mask, n_rounds)
    elif strategy_name == "lc":
        return LeastConfidence(unlabel_mask, n_rounds)
    elif strategy_name == "margin":
        return MarginSampling(unlabel_mask, n_rounds)
    elif strategy_name == "margin2":
        return MarginSampling2(unlabel_mask, n_rounds)
    elif strategy_name == "entropy":
        return EntropySampling(unlabel_mask, n_rounds)
    elif strategy_name == "kcenter":
        return KCenterGreedy(unlabel_mask, n_rounds)
    else:
        raise NotImplementedError
