import numpy as np


class ProtoTree:

    def _get_independent(self, data, target):
        """
        Derive independent attribute names and remove the target attribute string.

        data:   pandas.core.frame.DataFrame
        target: string

        -> return: list
        """
        return data.columns.tolist().remove(target)

    def _pass_through_tree(self, row):
        """
        Return frequency-based probability for a given list.

        row: list

        -> return: list
        """
        return self.data[self.target].value_counts().apply(lambda x: x/len(self.data)).tolist()

    def fit(self, data, target):
        """
        Derive and self-assign (training) data, target attribute, and independent attribute names.

        data:   pandas.core.frame.DataFrame
        target: string
        """

        self.data   = data
        self.target = target

        self.independent = self._get_independent(data, target)

    def predict(self, data):
        """
        Iterate through (test) data and return frequency-based probability.

        data: pandas dataframe

        -> return: numpy.ndarray
        """
        return np.array([self._pass_through_tree(row) for row in data.values])











