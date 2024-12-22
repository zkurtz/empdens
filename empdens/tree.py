"""Decision tree density estimation."""

import numpy as np

from empdens.base import AbstractDensity

# TODO: generalize to handle categoricals


def log_node_volume(node) -> float:
    """Compute the log-volume of a node in terms of its bounds."""
    diffs = node.bounds[1] - node.bounds[0]
    ldiffs = np.log(diffs)
    return np.sum(ldiffs)


def node_volume(node):
    """Compute the volume of a node in terms of its bounds."""
    lv = log_node_volume(node)
    return np.exp(lv)


def split_bounds(bounds, feature_idx, value) -> dict[str, np.ndarray]:
    """Split the bounds of a node along a feature.

    Args:
        bounds: Two rows, with upper and lower bounds for each feature.
        feature_idx: Index of feature to split on.
        value: Value to split on.
    """
    bounds_left = np.copy(bounds)
    bounds_right = np.copy(bounds)
    bounds_left[1, feature_idx] = value
    bounds_right[0, feature_idx] = value
    return {"left": bounds_left, "right": bounds_right}


class Node:
    """Store the attributes of a node.

    - bounds
    - if a leaf, the list of indices of training data contained
    - if not a leaf, the split that defines its children.
    """

    def __init__(self, bounds, members) -> None:
        """Initialize the node.

        :param bounds: (np.array) two rows, with upper and lower bounds for each feature
        :param members: (numpy integer array) indices of training data falling in this node
        """
        self.bounds = bounds
        self.members = members
        # until we decide to split it, this is simply a leaf
        self.leaf = True
        self.left = None
        self.right = None

    def split(
        self,
        feature_idx: int,
        value: float,
        left_members: np.ndarray,
        right_members: np.ndarray,
    ) -> None:
        """Split the node into two children.

        Args:
            feature_idx: (int) index of the feature to split on
            value: (float) value to split on
            left_members: (numpy integer array) indices of training data falling in the left child
            right_members: (numpy integer array) indices of training data falling in the right child
        """
        child_bounds = split_bounds(self.bounds, feature_idx, value)
        self.left = Node(bounds=child_bounds["left"], members=left_members)
        self.right = Node(bounds=child_bounds["right"], members=right_members)
        self.members = None
        self.leaf = False


class Tree:
    """A decision tree for density estimation."""

    def __init__(self, df):
        """Initialize the tree."""
        self.n_features = df.shape[1]
        self.n_points = df.shape[0]
        self._plant_me(df)
        while True:
            self._search_split(df)

    def _plant_me(self, df):
        bounds = np.array([df.min(axis=0), df.max(axis=0)])
        self.idx = list(range(self.n_points))
        self.root = Node(bounds, members=list(range(self.n_points)))
        self.leaves = [self.root]

    def _search_split(self, df):
        raise NotImplementedError("Not clear what the plan was for this.")


class TreeDensity(AbstractDensity):
    """A decision-tree density."""

    def __init__(self, verbose=False):
        """Initialize the decision tree density model."""
        super().__init__(verbose=verbose)

    def train(self, df):
        """Model the density of the data.

        :param df: (pandas DataFrame)
        """
        self._validate_data(df)
        self.vp("Training a decision tree density model on " + str(df.shape[0]) + " samples")
        self.tree = Tree(df)

    def density(self, X):
        """Predict the density at new points.

        Apply equation 2.1 in https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf

        :param X: (pd.DataFrame or numpy array) Must match the exact column order of the `df`
            argument that was passed to self.train
        """
        self._validate_data(X)
        raise NotImplementedError("Not clear what the plan was for this.")
