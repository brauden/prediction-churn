# Tabular data experiments

### Data
The data set used in this experiment: [news popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity).

### Run

The experiment has to be run from the root folder:
```shell
python churn/tabular/experiments_tabular.py
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

### Performance comparison
The result of the experiments on the tabular data averaged across 10 runs:

| Method       | Hyperparameters    | Accuracy | Churn | Good Churn | Bad Churn | Win-Loss Ratio | Churn Ratio |
|--------------|--------------------|----------|-------|------------|-----------|----------------|-------------|
| Baseline     | None               | 0.66     | 0.10  | 0.05       | 0.05      | 1.02           | 1           |
| Distillation | lambda=0.2         | 0.657    | 0.093 | 0.047      | 0.045     | 1.045          | 0.920       |
| Distillation | lambda=0.4         | 0.658    | 0.080 | 0.041      | 0.039     | 1.062          | 0.800       |
| Distillation | lambda=0.6         | 0.658    | 0.066 | 0.035      | 0.031     | 1.100          | 0.656       |
| Distillation | lambda=0.8         | 0.657    | 0.051 | 0.026      | 0.025     | 1.079          | 0.506       |
| Anchor       | alpha=0.2, eps=0.6 | 0.656    | 0.069 | 0.035      | 0.034     | 1.035          | 0.687       |
| Anchor       | alpha=0.2, eps=0.8 | 0.657    | 0.080 | 0.041      | 0.039     | 1.052          | 0.797       |
| Anchor       | alpha=0.2, eps=1.0 | 0.657    | 0.099 | 0.051      | 0.049     | 1.049          | 0.985       |
| Anchor       | alpha=0.4, eps=0.6 | 0.657    | 0.065 | 0.033      | 0.032     | 1.063          | 0.646       |
| Anchor       | alpha=0.4, eps=0.8 | 0.659    | 0.075 | 0.039      | 0.036     | 1.105          | 0.743       |
| Anchor       | alpha=0.4, eps=1.0 | 0.658    | 0.091 | 0.047      | 0.044     | 1.059          | 0.899       |
| Anchor       | alpha=0.6, eps=0.6 | 0.657    | 0.061 | 0.032      | 0.029     | 1.079          | 0.605       |
| Anchor       | alpha=0.6, eps=0.8 | 0.658    | 0.071 | 0.037      | 0.034     | 1.103          | 0.708       |
| Anchor       | alpha=0.6, eps=1.0 | 0.658    | 0.080 | 0.042      | 0.039     | 1.083          | 0.796       |
| Anchor       | alpha=0.8, eps=0.6 | 0.657    | 0.055 | 0.028      | 0.027     | 1.059          | 0.542       |
| Anchor       | alpha=0.8, eps=0.8 | 0.658    | 0.061 | 0.032      | 0.029     | 1.084          | 0.609       |
| Anchor       | alpha=0.8, eps=1.0 | 0.658    | 0.068 | 0.035      | 0.033     | 1.074          | 0.676       |

![](../../data/distillation-rcp.png)
![](../../data/anchor_heatmap.png)
