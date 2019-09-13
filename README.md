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

<details><summary>ex. train output</summary>

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

<details><summary>ex. test output</summary>

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
