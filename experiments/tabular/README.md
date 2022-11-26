# Tabular data experiments

### Data
The data set used in this experiment: [news popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity).

### Run

The experiment has to be run from the root folder:
```shell
python experiments/tabular/experiments_tabular.py
```

To open mlflow ui after the first run:
```shell
mlflow ui
```

### Structure
1. utils.py - helper functions to download the dataset from the internet
2. get_data.py - download the data
3. preprocessing.py - preprocess, split, and create torch Dataset
4. models.py - model used in the experiments. Simple FFN.
5. churn.py - churn reduction transformation: Knowledge distiallation and Anchor RCP, training procedure
6. experiments_tabular.py - run experiment and log results to mlflow