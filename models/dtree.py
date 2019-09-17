import numpy as np

from .base import ProtoTree

class DecisionTree(ProtoTree):

    def __init__(self, max_depth=4, depth=1):

        self.L = None
        self.R = None

        self.max_depth = max_depth
        self.depth     = depth

        self.criteria      = None
        self.split_feature = None

        self.impurity = 1

    def _gini_impurity(self, data):

        if data.empty:
            return 0

        p = data.value_counts().apply(lambda x: x/len(self.data)).tolist()

        # binary classification means two resultant probabilities whose sum is 1; p*(1-p) is identical for both cases
        # therefore, we can simply double the value of one case
        print(p)
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

    def _validate(self):

        non_numeric = self.data[self.independent].select_dtypes(
            include=['category', 'object', 'bool']
        ).columns.tolist()

        if len(set(self.independent)).intersection(set(non_numeric)) != 0:
            raise RuntimeError('all data features must be numeric')

        self.data[self.target] = self.data[self.target].astype('category')

        if len(self.data[self.target]).cat.categories != 2:
            raise RuntimeError('binary implementation only: data features must have <= 2 cases each')

    def fit(self, data, target):
        """
        Derive and self-assign (training) data, target attribute, and independent attribute names.

        data:   pandas.core.frame.DataFrame
        target: string
        """

        if self.depth <= self.max_depth:
            print(f'processing at depth: {self.depth}')

        self.data   = data
        self.target = target

        self.independent = self._get_independent(data, target)

        if self.depth <= self.max_depth:

            #self._validate()

            self.impurity = self._gini_impurity(self.data[self.target])

            self.criteria, self.split_feature, self.info_gain = self._best_split()

            if self.criteria is not None and self.info_gain > 0:
                self._branch()
        else:
            print('Branching ends; max depth has been reached')



