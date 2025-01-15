from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Optional
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class DataSplit:
    """Container for a single data split"""
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray



class RepeatedStratifiedGroupCV:
    """
    Creates repeated stratified group k-fold cross validation splits,
    falling back to random group assignment if a single-class test fold is encountered,
    and doing so for each of n_repeats.
    """

    def __init__(self,
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 random_state: Optional[int] = None,
                 max_fallback_attempts: int = 100):
        """
        Args:
            n_splits: Number of folds.
            n_repeats: Number of times to repeat CV.
            random_state: Random seed for reproducibility.
            max_fallback_attempts: Number of random attempts to find
                a valid fallback partition if single-class folds appear.
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.max_fallback_attempts = max_fallback_attempts
        self.splits_: List[Dict[str, np.ndarray]] = []


    def split(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Generate repeated stratified group splits, yielding (train_idx, test_idx).
        Also stores all folds in self.splits_.
        """
        rng = np.random.RandomState(self.random_state)
        self.splits_ = []

        # We'll do n_repeats. For each repeat:
        for _ in range(self.n_repeats):
            # 1) Do StratifiedGroupKFold for *this* repeat
            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=rng.randint(0, 2**31 - 1)
            )

            # Collect the folds for this single repeat
            standard_folds = []
            has_single_class_fold = False

            for train_idx, test_idx in cv.split(X, y, groups):
                test_classes = np.unique(y[test_idx])
                if len(test_classes) < 2:
                    has_single_class_fold = True
                fold_dict = {
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "train_groups": np.unique(groups[train_idx]),
                    "test_groups": np.unique(groups[test_idx])
                }
                standard_folds.append(fold_dict)

            if has_single_class_fold:
                # 2) Fallback: random assignment for this repeat
                fallback_folds = self._random_fallback_splits(
                    X, y, groups, rng=rng
                )
                # We'll get exactly n_splits folds from fallback
                for fold in fallback_folds:
                    self.splits_.append(fold)
                    yield fold['train_idx'], fold['test_idx']
            else:
                # 3) If no single-class test fold, use standard folds
                for fold in standard_folds:
                    self.splits_.append(fold)
                    yield fold['train_idx'], fold['test_idx']


    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns total number of folds = n_splits * n_repeats."""
        return self.n_splits * self.n_repeats


    def _random_fallback_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        rng: np.random.RandomState
    ) -> List[Dict[str, np.ndarray]]:
        """
        Attempt random group assignments to n_splits folds, ensuring each fold:
          - is non-empty
          - has >= 2 classes in test
        We'll try up to self.max_fallback_attempts times.
        Once we find a valid partition, we build and return exactly n_splits folds
        (each with train_idx, test_idx, etc.).
        """
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        for attempt in range(self.max_fallback_attempts):
            # Randomly assign each group to one of the folds
            assignment = rng.randint(0, self.n_splits, size=n_groups)

            # Check that no fold is empty
            fold_counts = np.bincount(assignment, minlength=self.n_splits)
            if np.any(fold_counts == 0):
                continue

            # Build fold->classes
            fold_to_classes = [set() for _ in range(self.n_splits)]
            for i, g in enumerate(unique_groups):
                fold_id = assignment[i]
                sample_idx = np.where(groups == g)[0]
                fold_to_classes[fold_id].update(y[sample_idx])

            # Each fold must have >=2 classes
            if all(len(classes_set) >= 2 for classes_set in fold_to_classes):
                # Build the actual folds
                folds_out = []
                for fold_id in range(self.n_splits):
                    test_groups = unique_groups[assignment == fold_id]
                    test_mask = np.isin(groups, test_groups)
                    train_mask = ~test_mask

                    train_idx = np.where(train_mask)[0]
                    test_idx = np.where(test_mask)[0]
                    folds_out.append({
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "train_groups": np.unique(groups[train_idx]),
                        "test_groups": np.unique(groups[test_idx])
                    })
                return folds_out

        # If we exit loop, no valid arrangement found
        raise RuntimeError(
            f"No valid fallback partition found after {self.max_fallback_attempts} attempts."
        )


