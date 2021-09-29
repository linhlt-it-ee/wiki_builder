import copy
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from torchmetrics.classification import F1

class Strategy:
    def __init__(self, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
        self.n_rounds = n_rounds
        self.unlabel_mask = unlabel_mask
        self.n_samples = len(self.unlabel_mask)
        self.train_mask = torch.zeros(self.n_samples, dtype=torch.bool)
        self.n_samples_per_round = sum(self.unlabel_mask).item() // n_rounds

    def update(self, select_idx):
        self.train_mask[select_idx] = True
        self.unlabel_mask[select_idx] = False

    def _random_idx(self) -> np.ndarray:
        train_idx = torch.where(self.unlabel_mask)[0].numpy()
        rand_idx = np.random.choice(train_idx, replace=False, size=self.n_samples_per_round)
        return rand_idx

    def random_mask(self) -> torch.BoolTensor:
        self.update(self._random_idx())
        return self.train_mask

    def query(self, probs: torch.Tensor = None, features: torch.Tensor = None) -> torch.BoolTensor:
        pass

class RandomSampling(Strategy):
    def query(
        self,
        probs: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        features: torch.Tensor = None
    ) -> torch.BoolTensor:
        return super().random_mask()

class LeastConfidence(Strategy):
    def query(
        self,
        probs: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        features: torch.Tensor = None
    ) -> torch.BoolTensor:
        confidence = probs[self.unlabel_mask].max(dim=1).values
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[confidence.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class EntropySampling(Strategy):
    def query(
        self,
        probs: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        features: torch.Tensor = None
    ) -> torch.BoolTensor:
        p = probs[self.unlabel_mask]
        neg_entropy = (p * p.log()).sum(dim=1)
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[neg_entropy.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class MarginSampling(Strategy):
    def query(
        self,
        probs: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        features: torch.Tensor = None
    ) -> torch.BoolTensor:
        top2_p = probs[self.unlabel_mask].sort(dim=1).values[:, -2:]
        margin = top2_p[:, 1] - top2_p[:, 0]
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[margin.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class MarginSampling2(Strategy):
    def query(
        self,
        probs: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        features: torch.Tensor = None
    ) -> torch.BoolTensor:
        margin = torch.abs(probs[self.unlabel_mask] - 0.5).min(dim=1).values
        unlabel_idx = torch.arange(self.n_samples)[self.unlabel_mask]
        select_idx = unlabel_idx[margin.sort().indices[:self.n_samples_per_round]]
        self.update(select_idx)
        return self.train_mask

class KCenterGreedy(Strategy):
    def query(
        self,
        probs: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        features: torch.Tensor = None
    ) -> torch.BoolTensor:
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

class FeatureMatching(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = []
        self.out = []

    def query(
        self,
        probs: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        features: torch.Tensor = None
    ) -> torch.BoolTensor:
        if not self.groups:
            train_mask, unlabel_mask = copy.deepcopy(self.train_mask), copy.deepcopy(self.unlabel_mask)
            for _ in range(self.n_rounds - 1):
                random_idx = self._random_idx()
                self.groups.append(random_idx)
                self.update(random_idx)
                self.out.append(False)
            self.train_mask, self.unlabel_mask = train_mask, unlabel_mask

        min_score, min_idx = None, None
        for i, batch_idx in enumerate(self.groups):
            if self.out[i]:
                continue
            score = F1(average="micro")(probs[batch_idx], labels[batch_idx]).item()
            if min_idx is None or score < min_score:
                min_idx = i
                min_score = score
        self.out[min_idx] = True
        self.update(self.groups[min_idx])
        return self.train_mask

def prepare_strategy(strategy_name: str, unlabel_mask: torch.BoolTensor, n_rounds: int = 100):
    strategy_dict = {
        "random": RandomSampling, "lc": LeastConfidence,
        "margin": MarginSampling, "margin2": MarginSampling2,
        "entropy": EntropySampling, "kcenter": KCenterGreedy,
        "fm": FeatureMatching,
    }
    return strategy_dict[strategy_name](unlabel_mask, n_rounds)
