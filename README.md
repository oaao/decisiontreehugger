# decisiontreehugger

:see_no_evil: what if we :flushed: learn about how ML algorithms work, not just `scikit-learn` about `instantiate`, `fit`, and `predict` :kissing_heart:

Examples use a preprocessed dataset of the [Titanic passenger manifest](./example_data/). Recreate the process or roll your own for other data by following the Decision Tree approach outlined in [this Kaggle companion notebook](https://www.kaggle.com/startupsci/titanic-data-science-solutions).

## usage

Start by establishing training and test data:

```python
>>> import pandas
>>>
>>> train = pandas.read_csv('./example_data/titanic_train_preprocessed.csv')
>>> test  = pandas.read_csv('./example_data/titanic_test_preprocessed.csv')
```

<details><summary>expand to view example train output</summary>

<p>

```python
>>> train
           Age     Fare  Embarked_C  Embarked_Q  Embarked_S  Cabin_A  Cabin_B  ...  Master  Miss  Mr  Mrs  Officer  Royalty  Survived
0    22.000000   7.2500           0           0           1        0        0  ...       0     0   1    0        0        0       0.0
1    38.000000  71.2833           1           0           0        0        0  ...       0     0   0    1        0        0       1.0
2    26.000000   7.9250           0           0           1        0        0  ...       0     1   0    0        0        0       1.0
3    35.000000  53.1000           0           0           1        0        0  ...       0     0   0    1        0        0       1.0
4    35.000000   8.0500           0           0           1        0        0  ...       0     0   1    0        0        0       0.0
..         ...      ...         ...         ...         ...      ...      ...  ...     ...   ...  ..  ...      ...      ...       ...
886  27.000000  13.0000           0           0           1        0        0  ...       0     0   0    0        1        0       0.0
887  19.000000  30.0000           0           0           1        0        1  ...       0     1   0    0        0        0       1.0
888  29.881138  23.4500           0           0           1        0        0  ...       0     1   0    0        0        0       0.0
889  26.000000  30.0000           1           0           0        0        0  ...       0     0   1    0        0        0       1.0
890  32.000000   7.7500           0           1           0        0        0  ...       0     0   1    0        0        0       0.0

[891 rows x 29 columns]
```

</p>
</details>

<details><summary>expand to view example test output</summary>

<p>

```python
>>> test
           Age      Fare  Embarked_C  Embarked_Q  Embarked_S  Cabin_A  Cabin_B  ...  Pclass_3  Master  Miss  Mr  Mrs  Officer  Royalty
0    34.500000    7.8292           0           1           0        0        0  ...         1       0     0   1    0        0        0
1    47.000000    7.0000           0           0           1        0        0  ...         1       0     0   0    1        0        0
2    62.000000    9.6875           0           1           0        0        0  ...         0       0     0   1    0        0        0
3    27.000000    8.6625           0           0           1        0        0  ...         1       0     0   1    0        0        0
4    22.000000   12.2875           0           0           1        0        0  ...         1       0     0   0    1        0        0
..         ...       ...         ...         ...         ...      ...      ...  ...       ...     ...   ...  ..  ...      ...      ...
413  29.881138    8.0500           0           0           1        0        0  ...         1       0     0   1    0        0        0
414  39.000000  108.9000           1           0           0        0        0  ...         0       0     0   0    0        0        1
415  38.500000    7.2500           0           0           1        0        0  ...         1       0     0   1    0        0        0
416  29.881138    8.0500           0           0           1        0        0  ...         1       0     0   1    0        0        0
417  29.881138   22.3583           1           0           0        0        0  ...         1       1     0   0    0        0        0

[418 rows x 28 columns]

```

</p>
</details>

To get a baseline sense of expected behaviour, let's look at boolean-case survival statistics for our training set:

```python
>>> passengers = train.Survived
>>> survived   = sum(p for p in passengers if p == 1.0)
>>>
>>> survived / len(passengers)
0.3838383838383838
```
### proto-model

The `ProtoTree` is the rudimentary proto-model for the `DecisionTree`. We aren't creating and evaluating a decision tree here, but rather verifying the foundational integrity of our process.

```python
>>> from models.base import ProtoTree
>>>
>>> dt = ProtoTree()
>>> dt.fit(data=train, target='Survived')
>>>
>>> # we expect the aforementioned survival rate, consistent to all rows:
>>> predictions = dt.predict(test)
>>> predictions[:3]
array([[0.61616162, 0.38383838],
       [0.61616162, 0.38383838],
       [0.61616162, 0.38383838]])
```

We receive the reassuring but useless projection of a `0.38383838` survival rate for our test data, indicating that the training data probabilities have been processed correctly.


### decision tree

<details><summary>Expand this to read context on the design parametrisation of the decision tree.</summary>

<p>

**properties:**

* contains a root node

* each node may have a left and right branch

* bottom-layer nodes do not have branches

**considerations:**

* prioritise the most 'efficient' conditions at the top of the tree

* branches can be recursively implemented decision trees rather than separately articulated

* how do we create the tree to have optimal splits?

    + *ideal split*: an *ideal* split in binary classification produces homogenous branches

    + *impurity*: homogeneity is unrealistic - how can we decrease impurity in child node w.r.t. parent node?

        - Gini impurity

        - cross-entropy / information gain (logarithmic calculation)

</p>
</details>

<details><summary>Expand this to read how splitting behaviour is decided by calculation methods for impurity/misclassifaction rate.</summary>
<p>


**Gini impurity:**

Randomly pick a "row" / data point; it has `k` value cases, each with a number of occurrences in the dataset. Based on the proportional distribution of a case and the `k` cases it could be randomly classified as, Gini impurity is the probability sum of all *incorrect* classification events.

For example, assume we have received the following dataset of `Yes`, `No`, and `Maybe` votes for some proposition. Since we're working with a binary classification problem, we'll use an example attribute/feature with `k=2` cases.

|           |   case   | count   | proportion |
|-----------|:--------:|:-------:|:----------:|
|           |  Yes     |   6     |     60%    |
|           |   No     |   4     |     40%    |
| **total** |  *(k=2)* |   *10*  |    *100%*  |

Here, `proportion` also represents `probability`; i.e. we have a 10% chance of randomly selecting a `Maybe` case.

Given the case distribution in our training data, we would e.g. randomly select `Yes` 60% of the time, and we would also randomly classify it as `Yes` 60% of the time. Therefore, the event probability of classifying `Yes` as `Yes` is `0.6 * 0.6 = 0.36`. The full series of classification event probabilities would be as follows:

| randomly selected | selection probability | randomly classified as | classification event probability | correctness |
|:-----------------:|-----------------------|:----------------------:|----------------------------------|:-----------:|
|        Yes        | 0.6                   |           Yes          | `0.6 * 0.6 =` 0.36               |      ✅      |
|        Yes        | 0.6                   |           No           | `0.6 * 0.4 =` 0.24               |      ❌      |
|         No        | 0.4                   |           Yes          | `0.4 * 0.6 =` 0.24               |      ❌      |
|         No        | 0.4                   |           No           | `0.4 * 0.4 =` 0.16               |      ✅      |

The *Gini impurity* would be the sum of all incorrect classification event probabilities:

`0.24 + 0.24` = **0.48**

Canonically, it is expressed by the following equation:

<img src="https://i.imgur.com/VzImiiu.png" />

> where `k` is the number of cases in the target attribute, and `p` is the probability of a given case existing at that node

So, let's test whether our event table and the Gini impurity equation yield the same result:

```
G = p(1)*(1 - p(1)) + p(2)*(1 - p(2))
  = (0.6)*(1 - 0.6) + (0.4)*(1 - 0.4)  = 0.48
```


**information gain:**

<img src="https://i.imgur.com/cpHogrY.png" />

> where `k` is the number of cases in the target attribute, `r` is the row count in the node, and `R` is the row count for the entire dataset

</details>

## references

* Victor Zhou (2019): [A Simple Explanation of Gini Impurity](https://victorzhou.com/blog/gini-impurity/)

* [@amro](https://stackoverflow.com/users/97160/amro) via SO (2009): [ans: What is "entropy and information gain?"](https://stackoverflow.com/a/1859910)