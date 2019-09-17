import numpy as np

from .base import ProtoTree

class DecisionTree(ProtoTree):

    def __init__(self, max_depth=4, depth=1):
        super(ProtoTree, self).__init__()

        self.L = None
        self.R = None

        self.max_depth = max_depth
        self.depth     = depth

        self.criteria      = None
        self.split_feature = None

        self.impurity_rate = 1

    def _gini_impurity(self, data):

        if data.empty or not data:
            return 0

        p = data.value_counts().apply(lambda x: x/len(self.data)).tolist()

        # binary classification means two resultant probabilities whose sum is 1; p*(1-p) is identical for both cases
        # therefore, we can simply double the value of one case
        return 2 * p * (1 - p)

    def _information_gain(self, feature, value):

            L = feature <= v
            R = feature >  v

            L_data, R_data = self.data[L], self.data[R]

            L_impurity = self._gini_impurity(L_data[self.target])
            R_impurity = self._gini_impurity(R_data[self.target])

            gain = self.impurity_rate \
                   - (len(L_data) / len(self.data)) * L_impurity \
                   + (len(R_data) / len(self.data)) * R_impurity

            return gain

    def _best_split_per_feature(self, attr):

        feature = self.data[attr]
        uniques = feature.unique()

        info_gain = None
        split     = None

        if len(uniques) == 1:
            return info_gain, split

        for value in uniques:
            potential_gain = self._information_gain(feature, value)

            if not info_gain or potential_gain > info_gain:
                info_gain = potential_gain
                split     = value

        return info_gain, split


    def _best_split(self):

        best = {}

        for feature in self.independent:

            info_gain, split = self._best_split_per_feature(feature)

            if not split:
                continue
            if not split or split['gain'] < info_gain:
                best = {'split': split, 'feature': feature, 'gain': info_gain}

        return best['split'], best['feature']

    def _branch(self):

        self.L = DecisionTree(max_depth=self.max_depth, depth=self.depth+1)
        self.R = DecisionTree(max_depth=self.max_depth, depth=self.depth+1)

        L_rows = self.data[self.data[self.split_feature] <= self.criteria]
        R_rows = self.data[self.data[self.split_feature] >  self.criteria]

        self.L.fit(data=L_rows, target=self.target)
        self.R.fit(data=R_rows, target=self.target)





