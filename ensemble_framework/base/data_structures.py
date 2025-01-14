from dataclasses import dataclass
import numpy as np
from typing import List, Optional
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class DataSplit:
    """Container for a single data split"""
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray


class RepeatedStratifiedGroupCV:
    """Creates repeated stratified group k-fold cross validation splits."""

    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> List[DataSplit]:
        """Generate repeated stratified group splits"""
        splits = []
        rng = np.random.RandomState(self.random_state)

        for repeat in range(self.n_repeats):
            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=rng.randint(0, 1000000)
            )

            for train_idx, test_idx in cv.split(X, y, groups):
                splits.append(DataSplit(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    train_groups=np.unique(groups[train_idx]),
                    test_groups=np.unique(groups[test_idx])
                ))

        return splits