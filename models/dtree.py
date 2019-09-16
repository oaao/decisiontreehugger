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
