import numpy as np

from .base import ProtoTree

class DecisionTree(ProtoTree):

    def __init__(self):
        super(ProtoTree, self).__init__()

        self.impurity_rate = 1

    def _gini_impurity(self, data):

        if data.empty or not data:
            return 0

        p = data.value_counts().apply(lambda x: x/len(self.data)).tolist()

        # binary classification means two resultant probabilities whose sum is 1; p*(1-p) is identical for both cases
        # therefore, we can simply double the value of one case
        return 2 * p * (1 - p)

    def _information_gain(self, feature, value):

            left  = feature <= v
            right = feature >  v

            l_data, r_data = self.data[left], self.data[right]

            l_impurity = self._gini_impurity(l_data[self.target])
            l_impurity = self._gini_impurity(r_data[self.target])

            gain = self.impurity_rate \
                   - (len(l_data) / len(self.data)) * l_impurity \
                   + (len(r_data) / len(self.data)) * r_impurity

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
