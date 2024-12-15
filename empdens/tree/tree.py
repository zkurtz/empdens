import pdb

import numpy as np

from empdens.base import AbstractDensity

# TODO: generalize to handle categoricals


def log_node_volume(node):
    """Compute the log-volume of a node in terms of its bounds."""
    assert isinstance(node, Cube)
    diffs = node.bounds[1] - node.bounds[0]
    ldiffs = np.log(diffs)
    return np.sum(ldiffs)


def node_volume(node):
    """Compute the volume of a node in terms of its bounds."""
    lv = log_node_volume(node)
    return np.exp(lv)


def split_bounds(bounds, feature_idx, value):
    """:param bounds: the bounds to split
    :param feature_idx: index of feature to split on
    :param value: value to split on
    :return: (left, right) bounds tuple
    """
    bounds_left = np.copy(bounds)
    bounds_right = np.copy(bounds)
    bounds_left[1, feature_idx] = value
    bounds_right[0, feature_idx] = value
    return {"left": bounds_left, "right": bounds_right}


class Node:
    """Store the attributes of a node:
    - bounds
    - if a leaf, the list of indices of training data contained
    - if not a leaf, the split that defines its children.
    """

    def __init__(self, bounds, members):
        """:param bounds: (np.array) two rows, with upper and lower bounds for each feature
        :param members: (numpy integer array) indices of training data falling in this node
        """
        self.bounds = bounds
        self.members = members
        # until we decide to split it, this is simply a leaf
        self.leaf = True
        self.left = None
        self.right = None

    def split(self, feature_idx, value, left_members, right_members):
        child_bounds = split_bounds(self.bounds, feature_idx, value)
        self.left = Node(bounds=child_bounds["left"], members=left_members)
        self.right = Node(bounds=child_bounds["right"], members=right_members)
        self.members = None
        self.leaf = False


class Tree:
    def __init__(self, df):
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
        # Initial density estimate
        pdb.set_trace()
